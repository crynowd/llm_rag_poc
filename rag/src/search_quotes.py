import argparse
import os
from typing import Any, Dict

import numpy as np
import faiss  # type: ignore

from utils import load_json, load_jsonl
from ollama_client import OllamaClient
from quote_extractor import extract_best_quote


def load_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    meta = {}
    for row in load_jsonl(meta_path):
        meta[row["chunk_id"]] = row
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Search query")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--max_quote_chars", type=int, default=None)
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    cfg = load_json(os.path.join(project_dir, "config.json"))
    embed_model = cfg["ollama"]["embedding_model"]
    k = int(args.k) if args.k is not None else int(cfg["retrieval"]["top_k"])
    max_quote_chars = (
        int(args.max_quote_chars)
        if args.max_quote_chars is not None
        else int(cfg["answering"]["max_quote_chars"])
    )

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
    qvec = client.embed_one(embed_model, args.query)
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)

    scores, idxs = index.search(q, k)

    print("\n" + "=" * 100)
    print(f"QUERY: {args.query}")
    print(f"MODEL: {embed_model} | top_k={k}")
    print("=" * 100)

    for rank, (i, score) in enumerate(zip(idxs[0], scores[0]), start=1):
        if i < 0 or i >= len(ids):
            continue

        chunk_id = ids[i]
        m = meta.get(chunk_id, {})
        full_text = chunk_text_by_id.get(chunk_id, "")

        where = []
        if m.get("page_min") is not None:
            where.append(f"pages {m.get('page_min')}–{m.get('page_max')}")
        if m.get("para_min") is not None:
            where.append(f"paras {m.get('para_min')}–{m.get('para_max')}")
        where_s = ", ".join(where) if where else "n/a"

        qc = extract_best_quote(
            chunk_text=full_text,
            query=args.query,
            max_quote_chars=max_quote_chars,
            window_sentences=2,
        )

        print("\n" + "-" * 100)
        print(f"[{rank}] score={score:.4f} | {chunk_id}")
        print(f"doc_id={m.get('doc_id')} | {where_s}")
        if qc.text:
            hits = ", ".join(qc.hit_words)
            print(f"quote_score={qc.score:.2f} | hits=[{hits}]")
            print("-" * 100)
            print(qc.text)
        else:
            print("quote_score=0.00 | hits=[]")
            print("-" * 100)
            print("(no matching quote found in this chunk)")

    print()


if __name__ == "__main__":
    main()
