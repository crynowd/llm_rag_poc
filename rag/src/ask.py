import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss  # type: ignore

from utils import load_json, load_jsonl, make_run_dir, setup_logger
from ollama_client import OllamaClient
from quote_extractor import extract_best_quote


def load_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(meta_path):
        meta[row["chunk_id"]] = row
    return meta


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


def build_prompt(query: str, quotes: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = (
        "Ты корпоративный ассистент по нормативным документам. "
        "Отвечай ТОЛЬКО на основе предоставленных цитат. "
        "Если ответа нет в цитатах — верни ровно: NOT_FOUND."
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


def retrieve(index: faiss.Index, ids: List[str], qvec: List[float], k: int) -> List[Tuple[str, float]]:
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    out: List[Tuple[str, float]] = []
    for i, score in zip(idxs[0], scores[0]):
        if i < 0 or i >= len(ids):
            continue
        out.append((ids[i], float(score)))
    return out


def save_md(path: str, query: str, answer: str, quotes: List[Dict[str, Any]], retrieved: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Ad-hoc RAG query")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    lines.append("```")
    lines.append(query)
    lines.append("```")
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append("```")
    lines.append(answer.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Quotes passed to LLM")
    lines.append("")
    if not quotes:
        lines.append("_No quotes_")
    else:
        for q in quotes:
            src = q.get("source", {})
            lines.append(
                f"- chunk_id=`{q.get('chunk_id')}` | doc_id=`{src.get('doc_id')}` | "
                f"pages=`{src.get('pages')}` | paras=`{src.get('paras')}` | "
                f"sim={q.get('similarity'):.4f} | quote_score={q.get('quote_score'):.2f}"
            )
            lines.append("")
            lines.append("```")
            lines.append((q.get("quote") or "").strip())
            lines.append("```")
            lines.append("")
    lines.append("## Retrieval top")
    lines.append("")
    for r in retrieved:
        lines.append(f"- {r['chunk_id']} | doc_id={r.get('doc_id')} | sim={r['similarity']:.4f} | where={r.get('where')}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--max_quotes", type=int, default=3)
    ap.add_argument("--min_similarity", type=float, default=None)
    ap.add_argument("--doc_id", type=str, default=None, help="Optional: restrict search to a doc_id")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    cfg = load_json(os.path.join(project_dir, "config.json"))
    top_k = int(args.k) if args.k is not None else int(cfg["retrieval"]["top_k"])
    min_similarity = float(args.min_similarity) if args.min_similarity is not None else float(cfg["answering"]["min_similarity"])
    max_quote_chars = int(cfg["answering"]["max_quote_chars"])

    run_dir = make_run_dir(project_dir)
    logger = setup_logger("rag_ask", os.path.join(run_dir, "ask.log"))
    logger.info(f"query={args.query}")

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

    client = OllamaClient()
    qvec = client.embed_one(cfg["ollama"]["embedding_model"], args.query)

    # retrieve more, then optional filter
    retrieved_all = retrieve(index, ids, qvec, max(top_k * 10, top_k))
    if args.doc_id:
        retrieved_all = [(cid, sc) for (cid, sc) in retrieved_all if meta.get(cid, {}).get("doc_id") == args.doc_id] or retrieved_all
    retrieved_all = retrieved_all[:top_k]

    retrieved_view: List[Dict[str, Any]] = []
    for ch_id, sim in retrieved_all:
        m = meta.get(ch_id, {})
        pages, paras = format_where(m)
        where = []
        if pages: where.append(f"pages {pages}")
        if paras: where.append(f"paras {paras}")
        retrieved_view.append({"chunk_id": ch_id, "similarity": sim, "doc_id": m.get("doc_id"), "where": ", ".join(where) if where else "n/a"})

    if not retrieved_all or retrieved_all[0][1] < min_similarity:
        answer = "NOT_FOUND"
        quotes: List[Dict[str, Any]] = []
    else:
        quotes = []
        for ch_id, sim in retrieved_all:
            txt = chunk_text_by_id.get(ch_id, "")
            qc = extract_best_quote(txt, args.query, max_quote_chars=max_quote_chars, window_sentences=2)
            if not qc.text:
                continue
            m = meta.get(ch_id, {})
            pages, paras = format_where(m)
            quotes.append({
                "chunk_id": ch_id,
                "similarity": float(sim),
                "quote_score": float(qc.score),
                "quote": qc.text,
                "source": {"doc_id": m.get("doc_id"), "chunk_id": ch_id, "pages": pages, "paras": paras},
            })

        quotes.sort(key=lambda x: (x["similarity"], x["quote_score"]), reverse=True)
        quotes = quotes[: max(1, int(args.max_quotes))]

        if not quotes:
            answer = "NOT_FOUND"
        else:
            system, prompt = build_prompt(args.query, quotes)
            answer = client.generate(
                model=cfg["ollama"]["llm_model"],
                prompt=prompt,
                system=system,
                temperature=0.1,
            ).strip()

    # сохранить отчёт
    save_md(os.path.join(run_dir, "ask.md"), args.query, answer, quotes, retrieved_view)
    print(answer)


if __name__ == "__main__":
    main()
