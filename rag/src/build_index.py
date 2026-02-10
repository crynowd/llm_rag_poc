import os
import time
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

import faiss  # type: ignore

from utils import ensure_dir, load_json, load_jsonl, make_run_dir, setup_logger, write_jsonl
from ollama_client import OllamaClient


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    run_dir = make_run_dir(project_dir)
    log_path = os.path.join(run_dir, "index.log")
    logger = setup_logger("rag_index", log_path)

    logger.info("=== INDEX BUILD START ===")

    cfg = load_json(os.path.join(project_dir, "config.json"))
    embed_model = cfg["ollama"]["embedding_model"]
    top_k_default = int(cfg["retrieval"]["top_k"])
    logger.info(f"embedding_model={embed_model}")
    logger.info(f"retrieval.top_k={top_k_default}")

    chunks_path = os.path.join(project_dir, "data", "chunks", "chunks.jsonl")
    chunks = load_jsonl(chunks_path)
    logger.info(f"chunks_count={len(chunks)}")

    # outputs
    index_dir = os.path.join(project_dir, "data", "index")
    ensure_dir(index_dir)

    meta_path = os.path.join(index_dir, "chunk_meta.jsonl")
    emb_path = os.path.join(index_dir, "embeddings.npy")
    ids_path = os.path.join(index_dir, "chunk_ids.txt")
    faiss_path = os.path.join(index_dir, "faiss.index")

    # if embeddings already exist, we can skip recompute
    if os.path.isfile(emb_path) and os.path.isfile(meta_path) and os.path.isfile(ids_path):
        logger.info("Found existing embeddings/meta; will rebuild FAISS from cached embeddings.")
        embeddings = np.load(emb_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            chunk_ids = [line.strip() for line in f if line.strip()]
    else:
        client = OllamaClient()

        chunk_ids: List[str] = []
        meta_rows: List[Dict[str, Any]] = []
        vecs: List[List[float]] = []

        t0 = time.time()
        for c in tqdm(chunks, desc="Embedding chunks"):
            text = c["text"]
            chunk_id = c["chunk_id"]

            # Лёгкий предохранитель: очень длинные тексты не пихаем целиком
            # (у нас всё <= 3500 chars, так что почти никогда не сработает)
            if len(text) > 6000:
                text = text[:6000]

            emb = client.embed_one(embed_model, text)
            vecs.append(emb)

            chunk_ids.append(chunk_id)
            meta_rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": c["doc_id"],
                    "doc_type": c["doc_type"],
                    "source_path": c["source_path"],
                    "page_min": c["page_min"],
                    "page_max": c["page_max"],
                    "para_min": c["para_min"],
                    "para_max": c["para_max"],
                    "char_len": c["char_len"],
                }
            )

        dt = time.time() - t0
        logger.info(f"embeddings_built_in_sec={dt:.2f}")

        embeddings = np.array(vecs, dtype=np.float32)
        logger.info(f"embeddings_shape={embeddings.shape}")

        # save caches
        np.save(emb_path, embeddings)
        write_jsonl(meta_path, meta_rows)
        with open(ids_path, "w", encoding="utf-8") as f:
            for cid in chunk_ids:
                f.write(cid + "\n")

        logger.info(f"[OK] Saved embeddings: {emb_path}")
        logger.info(f"[OK] Saved meta: {meta_path}")
        logger.info(f"[OK] Saved chunk_ids: {ids_path}")

    # build FAISS index (cosine via inner product + normalization)
    d = embeddings.shape[1]
    logger.info(f"vector_dim={d}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    faiss.write_index(index, faiss_path)
    logger.info(f"[OK] Saved FAISS index: {faiss_path}")
    logger.info(f"index_total_vectors={index.ntotal}")

    logger.info("=== INDEX BUILD END ===")
    print(f"[OK] Index built: {faiss_path}")
    print(f"[LOG] {log_path}")


if __name__ == "__main__":
    main()