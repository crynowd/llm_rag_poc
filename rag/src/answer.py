import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss  # type: ignore

from src.utils import load_json, load_jsonl, make_run_dir, setup_logger
from src.ollama_client import OllamaClient
from src.quote_extractor import extract_best_quote


def load_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    meta = {}
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

    # Формируем контекст: цитаты с идентификаторами
    blocks = []
    for i, q in enumerate(quotes, start=1):
        src = q["source"]
        blocks.append(
            f"[Q{i}] SOURCE: doc_id={src['doc_id']}; chunk_id={src['chunk_id']}; "
            f"pages={src.get('pages')}; paras={src.get('paras')}\n"
            f"QUOTE:\n{q['quote']}"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query")
    ap.add_argument("--k", type=int, default=None, help="top_k retrieval")
    ap.add_argument("--max_quotes", type=int, default=3, help="how many quotes to pass to LLM")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    run_dir = make_run_dir(project_dir)
    log_path = os.path.join(run_dir, "ask.log")
    logger = setup_logger("rag_answer", log_path)

    cfg = load_json(os.path.join(project_dir, "config.json"))
    embed_model = cfg["ollama"]["embedding_model"]
    llm_model = cfg["ollama"]["llm_model"]
    top_k = int(args.k) if args.k is not None else int(cfg["retrieval"]["top_k"])
    min_similarity = float(cfg["answering"]["min_similarity"])
    max_quote_chars = int(cfg["answering"]["max_quote_chars"])
    not_found_policy = cfg["answering"]["not_found_policy"]

    logger.info("=== ANSWER START ===")
    logger.info(f"query={args.query}")
    logger.info(f"embedding_model={embed_model}")
    logger.info(f"llm_model={llm_model}")
    logger.info(f"top_k={top_k}")
    logger.info(f"min_similarity={min_similarity}")
    logger.info(f"max_quote_chars={max_quote_chars}")
    logger.info(f"not_found_policy={not_found_policy}")

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

    # embed query
    qvec = client.embed_one(embed_model, args.query)
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)

    scores, idxs = index.search(q, top_k)

    retrieved: List[Tuple[str, float]] = []
    for i, score in zip(idxs[0], scores[0]):
        if i < 0 or i >= len(ids):
            continue
        retrieved.append((ids[i], float(score)))

    logger.info(f"retrieved_count={len(retrieved)}")
    if retrieved:
        logger.info(f"best_score={retrieved[0][1]:.4f}")

    # NOT_FOUND if retrieval too weak
    if not retrieved or retrieved[0][1] < min_similarity:
        logger.info("NOT_FOUND due to low retrieval similarity.")
        print("NOT_FOUND")
        print(f"[LOG] {log_path}")
        return

    # extract quotes from retrieved chunks
    quotes: List[Dict[str, Any]] = []
    for chunk_id, sim in retrieved:
        full_text = chunk_text_by_id.get(chunk_id, "")
        qc = extract_best_quote(
            chunk_text=full_text,
            query=args.query,
            max_quote_chars=max_quote_chars,
            window_sentences=2,
        )
        if qc.text:
            m = meta.get(chunk_id, {})
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

            quotes.append(
                {
                    "chunk_id": chunk_id,
                    "similarity": sim,
                    "quote_score": qc.score,
                    "hit_words": qc.hit_words,
                    "quote": qc.text,
                    "source": {
                        "doc_id": m.get("doc_id"),
                        "chunk_id": chunk_id,
                        "pages": pages,
                        "paras": paras,
                    },
                }
            )

    logger.info(f"quotes_found={len(quotes)}")

    # NOT_FOUND if no quote inside retrieved
    if not quotes:
        logger.info("NOT_FOUND due to no quotes extracted from retrieved chunks.")
        print("NOT_FOUND")
        print(f"[LOG] {log_path}")
        return

    # pick best quotes: by similarity first, then quote_score
    quotes.sort(key=lambda x: (x["similarity"], x["quote_score"]), reverse=True)
    quotes = quotes[: max(1, int(args.max_quotes))]

    for qd in quotes:
        logger.info(
            f"quote chunk_id={qd['chunk_id']} sim={qd['similarity']:.4f} "
            f"quote_score={qd['quote_score']:.2f} hits={qd['hit_words']}"
        )

    system, prompt = build_prompt(args.query, quotes)

    answer = client.generate(
        model=llm_model,
        prompt=prompt,
        system=system,
        temperature=0.1,
    ).strip()

    logger.info("LLM answered.")
    logger.info(f"answer_preview={answer[:200].replace(chr(10), ' ')}")

    # Print result in a readable way (and keep sources visible)
    print("\n" + "=" * 100)
    print(f"ВОПРОС: {args.query}")
    print("=" * 100)
    print(answer)
    print("\n" + "-" * 100)
    print("ИСТОЧНИКИ (цитаты, переданные в LLM):")
    for i, qd in enumerate(quotes, start=1):
        src = qd["source"]
        print(
            f"[Q{i}] doc_id={src['doc_id']} | chunk_id={src['chunk_id']} | pages={src['pages']} | paras={src['paras']} | sim={qd['similarity']:.4f}"
        )
        print(qd["quote"])
        print()

    print(f"[LOG] {log_path}")
    logger.info("=== ANSWER END ===")


if __name__ == "__main__":
    main()
