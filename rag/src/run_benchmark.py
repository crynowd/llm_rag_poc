import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss  # type: ignore

from utils import load_json, load_jsonl, write_jsonl, make_run_dir, setup_logger
from ollama_client import OllamaClient
from quote_extractor import extract_best_quote


def load_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(meta_path):
        meta[row["chunk_id"]] = row
    return meta


def build_prompt(query: str, quotes: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Возвращает (system, user_prompt).
    """
    system = (
        "Ты корпоративный ассистент по нормативным документам. "
        "Ты ОБЯЗАН отвечать ТОЛЬКО на основе предоставленных цитат. "
        "НЕЛЬЗЯ добавлять новые факты, интерпретации, предположения. "
        "Если ответа нет в цитатах, верни ровно: NOT_FOUND. "
        "Не смешивай документы и редакции, используй только то, что дано."
    )

    blocks = []
    for i, q in enumerate(quotes, start=1):
        src = q["source"]
        blocks.append(
            f"[Q{i}] SOURCE: doc_id={src.get('doc_id')}; chunk_id={src.get('chunk_id')}; "
            f"pages={src.get('pages')}; paras={src.get('paras')}\n"
            f"QUOTE:\n{q.get('quote','')}"
        )

    context = "\n\n".join(blocks)

    user = f"""ВОПРОС:
{query}

ЦИТАТЫ (единственный источник истины):
{context}

ЗАДАНИЕ:
1) Дай краткий ответ на русском.
2) В конце укажи, какие цитаты использовал: [Q1], [Q2]...
3) Если в цитатах нет прямого ответа на вопрос, верни ровно: NOT_FOUND.
Формат:
- Ответ: ...
- Основание: [Q...]
"""
    return system, user


def normalize_text(s: str) -> str:
    # Для логов/сравнений: убираем лишние пробелы
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
    lines.append("## Quotes passed to LLM")
    lines.append("")
    quotes = payload.get("quotes", [])
    if not quotes:
        lines.append("_No quotes_")
    else:
        for q in quotes:
            src = q.get("source", {})
            lines.append(
                f"- **chunk_id:** `{q.get('chunk_id')}` | **doc_id:** `{src.get('doc_id')}` | "
                f"**pages:** `{src.get('pages')}` | **paras:** `{src.get('paras')}` | "
                f"**sim:** {q.get('similarity'):.4f} | **quote_score:** {q.get('quote_score'):.2f}"
            )
            lines.append("")
            lines.append("```")
            lines.append(q.get("quote", "").strip())
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
    ap.add_argument("--k", type=int, default=None, help="top_k retrieval (override config)")
    ap.add_argument("--max_quotes", type=int, default=3, help="How many quotes to pass to LLM")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases to run")
    ap.add_argument(
        "--restrict_doc",
        action="store_true",
        help="Restrict retrieved chunks to case.doc_id (recommended for this benchmark)",
    )
    ap.add_argument(
        "--no_restrict_doc",
        action="store_true",
        help="Disable doc restriction even if --restrict_doc not set",
    )
    ap.add_argument(
        "--k_search_multiplier",
        type=int,
        default=10,
        help="Retrieve k*k_search_multiplier candidates from FAISS before filtering",
    )
    ap.add_argument(
        "--min_similarity",
        type=float,
        default=None,
        help="Override answering.min_similarity from config",
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
    max_quote_chars = int(cfg["answering"]["max_quote_chars"])

    restrict_doc = False
    if args.no_restrict_doc:
        restrict_doc = False
    elif args.restrict_doc:
        restrict_doc = True
    else:
        # дефолт: без ограничения, чтобы поведение соответствовало твоему текущему пайплайну
        restrict_doc = False

    logger.info("=== BENCHMARK RUN START ===")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"cases_path={args.cases}")
    logger.info(f"embed_model={embed_model} llm_model={llm_model}")
    logger.info(f"top_k={top_k} max_quotes={args.max_quotes}")
    logger.info(f"min_similarity={min_similarity} max_quote_chars={max_quote_chars}")
    logger.info(f"restrict_doc={restrict_doc}")

    # snapshot config + args
    with open(os.path.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": args.cases,
                "k": top_k,
                "max_quotes": args.max_quotes,
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

    # Precompute chunk_id -> doc_id for filtering
    chunk_doc: Dict[str, Optional[str]] = {cid: meta.get(cid, {}).get("doc_id") for cid in ids}

    # Load cases
    cases = load_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]

    # init Ollama
    client = OllamaClient()

    out_rows: List[Dict[str, Any]] = []
    md_dir = os.path.join(run_dir, "cases_md")
    os.makedirs(md_dir, exist_ok=True)

    k_search = min(len(ids), max(top_k * int(args.k_search_multiplier), top_k))

    logger.info(f"cases_total={len(cases)} k_search={k_search}")

    for n, case in enumerate(cases, start=1):
        cid = case.get("id", f"case_{n:04d}")
        query = str(case.get("query", "")).strip()
        target_doc_id = case.get("doc_id")
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
            "quotes": [],
            "answer": "",
            "status": "OK",
            "notes": [],
        }

        try:
            # Embed query
            qvec = client.embed_one(embed_model, query)

            # Retrieve candidates
            retrieved_all = retrieve_candidates(index, ids, qvec, k_search)

            # Optional filter by doc_id (benchmark-focused)
            retrieved = retrieved_all
            if restrict_doc and target_doc_id:
                filtered = [(ch, sc) for (ch, sc) in retrieved_all if chunk_doc.get(ch) == target_doc_id]
                if filtered:
                    retrieved = filtered
                else:
                    case_payload["notes"].append("No chunks after doc_id filter; falling back to unfiltered.")
                    retrieved = retrieved_all

            retrieved = retrieved[:top_k]

            # Save retrieval info
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

            # NOT_FOUND if retrieval too weak
            if not retrieved or retrieved[0][1] < min_similarity:
                case_payload["answer"] = "NOT_FOUND"
                case_payload["status"] = "NOT_FOUND_LOW_SIM"
                out_rows.append(case_payload)
                save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
                continue

            # Extract quotes
            quotes: List[Dict[str, Any]] = []
            for ch_id, sim in retrieved:
                full_text = chunk_text_by_id.get(ch_id, "")
                qc = extract_best_quote(
                    chunk_text=full_text,
                    query=query,
                    max_quote_chars=max_quote_chars,
                    window_sentences=2,
                )
                if not qc.text:
                    continue

                m = meta.get(ch_id, {})
                pages, paras = format_where(m)

                quotes.append(
                    {
                        "chunk_id": ch_id,
                        "similarity": float(sim),
                        "quote_score": float(qc.score),
                        "hit_words": list(qc.hit_words),
                        "quote": qc.text,
                        "source": {
                            "doc_id": m.get("doc_id"),
                            "chunk_id": ch_id,
                            "pages": pages,
                            "paras": paras,
                        },
                    }
                )

            if not quotes:
                case_payload["answer"] = "NOT_FOUND"
                case_payload["status"] = "NOT_FOUND_NO_QUOTES"
                out_rows.append(case_payload)
                save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
                continue

            # Pick best quotes: by similarity then quote_score
            quotes.sort(key=lambda x: (x["similarity"], x["quote_score"]), reverse=True)
            quotes = quotes[: max(1, int(args.max_quotes))]

            case_payload["quotes"] = quotes

            # Generate answer
            system, prompt = build_prompt(query, quotes)
            answer = client.generate(
                model=llm_model,
                prompt=prompt,
                system=system,
                temperature=0.1,
            ).strip()

            case_payload["answer"] = answer

            # Minimal format sanity checks (только пометки, не ломаем прогон)
            a_norm = normalize_text(answer)
            if a_norm != "NOT_FOUND":
                if "Основание:" not in answer:
                    case_payload["notes"].append("Answer missing 'Основание:' line (format drift).")
                if "[Q" not in answer:
                    case_payload["notes"].append("Answer does not reference [Q...] (format drift).")

            out_rows.append(case_payload)
            save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)

        except Exception as e:
            case_payload["status"] = "ERROR"
            case_payload["answer"] = "NOT_FOUND"
            case_payload["notes"].append(f"Exception: {repr(e)}")
            out_rows.append(case_payload)
            save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
            logger.exception(f"case_id={cid} failed")

    # Save outputs
    results_path = os.path.join(run_dir, "results.jsonl")
    write_jsonl(results_path, out_rows)

    # A compact summary
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
