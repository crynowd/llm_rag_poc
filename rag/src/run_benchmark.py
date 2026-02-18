import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss  # type: ignore

from utils import load_json, load_jsonl, write_jsonl, make_run_dir, setup_logger
from ollama_client import OllamaClient


def load_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(meta_path):
        meta[row["chunk_id"]] = row
    return meta


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Возвращает (system, user_prompt).
    Передаём в LLM НЕ "цитаты", а найденные ЧАНКИ (или их обрезанную часть).
    """
    system = (
        "Ты корпоративный ассистент по нормативным документам. "
        "Ты ОБЯЗАН отвечать ТОЛЬКО на основе предоставленного контекста. "
        "НЕЛЬЗЯ добавлять новые факты, интерпретации, предположения. "
        "Если ответа нет в контексте, верни ровно: NOT_FOUND. "
        "Не смешивай документы и редакции, используй только то, что дано."
    )

    blocks: List[str] = []
    for i, c in enumerate(chunks, start=1):
        src = c["source"]
        blocks.append(
            f"[C{i}] SOURCE: doc_id={src.get('doc_id')}; chunk_id={src.get('chunk_id')}; "
            f"pages={src.get('pages')}; paras={src.get('paras')}\n"
            f"CHUNK:\n{c.get('text','')}"
        )

    context = "\n\n".join(blocks)

    user = f"""ВОПРОС:
{query}

КОНТЕКСТ (единственный источник истины):
{context}

ЗАДАНИЕ:
1) Дай краткий ответ на русском.
2) В конце укажи, какие фрагменты использовал: [C1], [C2]...
3) Если в контексте нет прямого ответа на вопрос, верни ровно: NOT_FOUND.
Формат:
- Ответ: ...
- Основание: [C...]
"""
    return system, user


def normalize_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def retrieve_candidates(
    index: faiss.Index,
    ids: List[str],
    qvec: List[float],
    k_search: int,
) -> List[Tuple[str, float]]:
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k_search)
    out: List[Tuple[str, float]] = []
    for i, score in zip(idxs[0], scores[0]):
        if i < 0 or i >= len(ids):
            continue
        out.append((ids[i], float(score)))
    return out


def format_where(m: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    pages = (
        f"{m.get('page_min')}–{m.get('page_max')}"
        if m.get("page_min") is not None
        else None
    )
    paras = (
        f"{m.get('para_min')}–{m.get('para_max')}"
        if m.get("para_min") is not None
        else None
    )
    return pages, paras


def save_case_markdown(path: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append(f"# {payload.get('id')}")
    lines.append("")
    lines.append(f"**Query:** {payload.get('query')}")
    lines.append(f"**Target doc_id:** {payload.get('doc_id')}")
    lines.append(f"**expect_not_found:** {payload.get('expect_not_found')}")
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append("```")
    lines.append(payload.get("answer", ""))
    lines.append("```")
    lines.append("")
    lines.append("## Chunks passed to LLM")
    lines.append("")
    chunks = payload.get("chunks", [])
    if not chunks:
        lines.append("_No chunks_")
    else:
        for c in chunks:
            src = c.get("source", {})
            lines.append(
                f"- **chunk_id:** `{c.get('chunk_id')}` | **doc_id:** `{src.get('doc_id')}` | "
                f"**pages:** `{src.get('pages')}` | **paras:** `{src.get('paras')}` | "
                f"**sim:** {c.get('similarity'):.4f}"
            )
            lines.append("")
            lines.append("```")
            lines.append((c.get("text", "") or "").strip())
            lines.append("```")
            lines.append("")
    lines.append("## Retrieval (top)")
    lines.append("")
    retr = payload.get("retrieved", [])
    if not retr:
        lines.append("_No retrieved chunks_")
    else:
        for r in retr:
            lines.append(
                f"- `{r.get('chunk_id')}` | doc_id=`{r.get('doc_id')}` | sim={r.get('similarity'):.4f} | where={r.get('where')}"
            )

    lines.append("")
    lines.append("## Gold (raw)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("gold", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Gold evidence (raw)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("gold_evidence", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="Path to cases.jsonl")
    ap.add_argument(
        "--doc_id",
        default=None,
        help="Restrict retrieval to a single document doc_id for ALL cases (variant A). "
             "If set, overrides case.doc_id and enables restriction automatically.",
    )
    ap.add_argument("--k", type=int, default=None, help="top_k retrieval (override config)")
    ap.add_argument(
        "--max_chunks",
        type=int,
        default=5,
        help="How many chunks to pass to LLM (replaces max_quotes / quote_extractor).",
    )
    ap.add_argument(
        "--max_chunk_chars",
        type=int,
        default=None,
        help="Trim each chunk to at most this many characters before passing to LLM "
             "(override answering.max_quote_chars if present).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases to run")
    ap.add_argument(
        "--restrict_doc",
        action="store_true",
        help="Restrict retrieved chunks to case.doc_id (or --doc_id override).",
    )
    ap.add_argument(
        "--no_restrict_doc",
        action="store_true",
        help="Disable doc restriction even if --restrict_doc not set",
    )
    ap.add_argument(
        "--k_search_multiplier",
        type=int,
        default=50,
        help="Retrieve k*k_search_multiplier candidates from FAISS before filtering (default raised).",
    )
    ap.add_argument(
        "--min_similarity",
        type=float,
        default=None,
        help="Override answering.min_similarity from config (used only as a weak gate).",
    )
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    run_dir = make_run_dir(project_dir)
    log_path = os.path.join(run_dir, "benchmark.log")
    logger = setup_logger("rag_benchmark", log_path)

    cfg_path = os.path.join(project_dir, "config.json")
    cfg = load_json(cfg_path)

    embed_model = cfg["ollama"]["embedding_model"]
    llm_model = cfg["ollama"]["llm_model"]
    top_k = int(args.k) if args.k is not None else int(cfg["retrieval"]["top_k"])
    min_similarity = (
        float(args.min_similarity)
        if args.min_similarity is not None
        else float(cfg["answering"]["min_similarity"])
    )
    # Reuse existing config param as default trimming budget
    cfg_trim = int(cfg.get("answering", {}).get("max_quote_chars", 1200))
    max_chunk_chars = int(args.max_chunk_chars) if args.max_chunk_chars is not None else cfg_trim

    restrict_doc = False
    if args.doc_id:
        restrict_doc = True
    elif args.no_restrict_doc:
        restrict_doc = False
    elif args.restrict_doc:
        restrict_doc = True
    else:
        restrict_doc = False

    logger.info("=== BENCHMARK RUN START ===")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"cases_path={args.cases}")
    logger.info(f"doc_id_override={args.doc_id}")
    logger.info(f"embed_model={embed_model} llm_model={llm_model}")
    logger.info(f"top_k={top_k} max_chunks={args.max_chunks}")
    logger.info(f"min_similarity={min_similarity} max_chunk_chars={max_chunk_chars}")
    logger.info(f"restrict_doc={restrict_doc} k_search_multiplier={args.k_search_multiplier}")

    with open(os.path.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": args.cases,
                "doc_id": args.doc_id,
                "k": top_k,
                "max_chunks": args.max_chunks,
                "max_chunk_chars": max_chunk_chars,
                "limit": args.limit,
                "restrict_doc": restrict_doc,
                "k_search_multiplier": args.k_search_multiplier,
                "min_similarity": min_similarity,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(run_dir, "config.snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Load index + chunks
    index_dir = os.path.join(project_dir, "data", "index")
    chunks_path = os.path.join(project_dir, "data", "chunks", "chunks.jsonl")
    meta_path = os.path.join(index_dir, "chunk_meta.jsonl")
    ids_path = os.path.join(index_dir, "chunk_ids.txt")
    faiss_path = os.path.join(index_dir, "faiss.index")

    chunks = load_jsonl(chunks_path)
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}

    meta = load_meta(meta_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    index = faiss.read_index(faiss_path)

    chunk_doc: Dict[str, Optional[str]] = {cid: meta.get(cid, {}).get("doc_id") for cid in ids}

    cases = load_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]

    client = OllamaClient()

    out_rows: List[Dict[str, Any]] = []
    md_dir = os.path.join(run_dir, "cases_md")
    os.makedirs(md_dir, exist_ok=True)

    # Pull lots of candidates before filtering (esp. for restrict_doc)
    k_search = min(len(ids), max(top_k * int(args.k_search_multiplier), 500 if restrict_doc else top_k))
    logger.info(f"cases_total={len(cases)} k_search={k_search}")

    for n, case in enumerate(cases, start=1):
        cid = case.get("id", f"case_{n:04d}")
        query = str(case.get("query", "")).strip()
        target_doc_id = args.doc_id if args.doc_id else case.get("doc_id")
        expect_nf = bool(case.get("expect_not_found", False))

        logger.info(f"[{n}/{len(cases)}] case_id={cid} target_doc_id={target_doc_id}")

        case_payload: Dict[str, Any] = {
            "id": cid,
            "query": query,
            "doc_id": target_doc_id,
            "expect_not_found": expect_nf,
            "gold": case.get("gold", {}),
            "gold_evidence": case.get("gold_evidence", {}),
            "retrieved": [],
            "chunks": [],
            "answer": "",
            "status": "OK",
            "notes": [],
        }

        try:
            qvec = client.embed_one(embed_model, query)

            retrieved_all = retrieve_candidates(index, ids, qvec, k_search)

            retrieved = retrieved_all
            if restrict_doc and target_doc_id:
                retrieved = [(ch, sc) for (ch, sc) in retrieved_all if chunk_doc.get(ch) == target_doc_id]
                if not retrieved:
                    case_payload["notes"].append("No chunks after doc_id filter (strict restrict_doc).")

            retrieved = retrieved[:top_k]

            for ch_id, sim in retrieved:
                m = meta.get(ch_id, {})
                pages, paras = format_where(m)
                where = []
                if pages:
                    where.append(f"pages {pages}")
                if paras:
                    where.append(f"paras {paras}")
                case_payload["retrieved"].append(
                    {
                        "chunk_id": ch_id,
                        "similarity": sim,
                        "doc_id": m.get("doc_id"),
                        "where": ", ".join(where) if where else "n/a",
                    }
                )

            if not retrieved or retrieved[0][1] < min_similarity:
                case_payload["answer"] = "NOT_FOUND"
                case_payload["status"] = "NOT_FOUND_EMPTY_RETRIEVAL"
                out_rows.append(case_payload)
                save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
                continue

            # Pass full chunks (trimmed) to LLM, NO quote_extractor
            chunks_for_llm: List[Dict[str, Any]] = []
            for ch_id, sim in retrieved[: max(1, int(args.max_chunks))]:
                full_text = chunk_text_by_id.get(ch_id, "") or ""
                text = full_text.strip()
                if max_chunk_chars and len(text) > max_chunk_chars:
                    text = text[:max_chunk_chars].rstrip() + "\n…"
                m = meta.get(ch_id, {})
                pages, paras = format_where(m)
                chunks_for_llm.append(
                    {
                        "chunk_id": ch_id,
                        "similarity": float(sim),
                        "text": text,
                        "source": {
                            "doc_id": m.get("doc_id"),
                            "chunk_id": ch_id,
                            "pages": pages,
                            "paras": paras,
                        },
                    }
                )

            if not chunks_for_llm:
                case_payload["answer"] = "NOT_FOUND"
                case_payload["status"] = "NOT_FOUND_NO_CHUNKS"
                out_rows.append(case_payload)
                save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
                continue

            case_payload["chunks"] = chunks_for_llm

            system, prompt = build_prompt(query, chunks_for_llm)
            answer = client.generate(
                model=llm_model,
                prompt=prompt,
                system=system,
                temperature=0.1,
            ).strip()

            case_payload["answer"] = answer

            a_norm = normalize_text(answer)
            if a_norm != "NOT_FOUND":
                if "Основание:" not in answer:
                    case_payload["notes"].append("Answer missing 'Основание:' line (format drift).")
                if "[C" not in answer:
                    case_payload["notes"].append("Answer does not reference [C...] (format drift).")

            out_rows.append(case_payload)
            save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)

        except Exception as e:
            case_payload["status"] = "ERROR"
            case_payload["answer"] = "NOT_FOUND"
            case_payload["notes"].append(f"Exception: {repr(e)}")
            out_rows.append(case_payload)
            save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
            logger.exception(f"case_id={cid} failed")

    results_path = os.path.join(run_dir, "results.jsonl")
    write_jsonl(results_path, out_rows)

    total = len(out_rows)
    n_ok = sum(1 for r in out_rows if r.get("status") == "OK")
    n_nf = sum(1 for r in out_rows if str(r.get("answer", "")).strip() == "NOT_FOUND")
    n_err = sum(1 for r in out_rows if r.get("status") == "ERROR")

    summary = {
        "total": total,
        "ok": n_ok,
        "not_found_answers": n_nf,
        "errors": n_err,
        "run_dir": run_dir,
        "results_jsonl": results_path,
        "cases_md_dir": md_dir,
        "log_path": log_path,
    }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=== BENCHMARK RUN END ===")
    logger.info(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
