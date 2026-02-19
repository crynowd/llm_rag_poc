import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss  # type: ignore
import re
from collections import Counter, defaultdict
import math
from utils import load_json, load_jsonl, write_jsonl, make_run_dir, setup_logger
from ollama_client import OllamaClient
from quote_extractor import extract_best_quote

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

def bm25_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower().replace("ё", "е"))

def bm25_rerank(
    query: str,
    candidates: List[Tuple[str, float]],
    chunk_text_by_id: Dict[str, str],
    k: int,
    alpha: float = 0.65,  # 0..1: 1 = только bm25, 0 = только similarity
) -> List[Tuple[str, float]]:
    """
    Возвращает список (chunk_id, combined_score) длиной k.
    candidates: (chunk_id, faiss_similarity)
    """
    if not candidates:
        return []

    q_tokens = bm25_tokenize(query)
    if not q_tokens:
        # нечем ранжировать по словам, оставляем как есть
        return candidates[:k]

    # строим "документы" на лету только для кандидатов
    doc_tokens: List[List[str]] = []
    doc_ids: List[str] = []
    doc_lens: List[int] = []

    for cid, sim in candidates:
        toks = bm25_tokenize(chunk_text_by_id.get(cid, ""))
        doc_tokens.append(toks)
        doc_ids.append(cid)
        doc_lens.append(len(toks))

    N = len(doc_ids)
    avgdl = (sum(doc_lens) / N) if N else 0.0
    if avgdl == 0:
        return candidates[:k]

    # df(term)
    df = defaultdict(int)
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1

    # idf(term)
    idf = {}
    for t, dft in df.items():
        # стандартная BM25 idf
        idf[t] = math.log(1 + (N - dft + 0.5) / (dft + 0.5))

    # BM25 params
    k1 = 1.2
    b = 0.75

    # скоринг
    bm25_scores = []
    for toks, dl in zip(doc_tokens, doc_lens):
        tf = Counter(toks)
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            f = tf[t]
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score += idf.get(t, 0.0) * (f * (k1 + 1) / denom)
        bm25_scores.append(score)

    # нормализация обоих скорингов в 0..1
    sims = [sim for _, sim in candidates]
    sim_min, sim_max = min(sims), max(sims)
    bm_min, bm_max = min(bm25_scores), max(bm25_scores)

    def norm(x, mn, mx):
        if mx <= mn:
            return 0.0
        return (x - mn) / (mx - mn)

    combined = []
    for (cid, sim), bm in zip(candidates, bm25_scores):
        s1 = norm(sim, sim_min, sim_max)
        s2 = norm(bm, bm_min, bm_max)
        combined_score = alpha * s2 + (1 - alpha) * s1
        combined.append((cid, combined_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]

def mmr_select(
    candidates: List[Dict[str, Any]],
    qvec: List[float],
    emb: np.ndarray,
    id_to_pos: Dict[str, int],
    max_select: int,
    lambda_mult: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance selection to diversify contexts.
    Expects candidates sorted by relevance (higher is better).
    Each candidate dict must contain: chunk_id, similarity (cosine).
    """
    if not candidates:
        return []

    q = np.array(qvec, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    selected: List[Dict[str, Any]] = []
    selected_vecs: List[np.ndarray] = []

    for _ in range(min(max_select, len(candidates))):
        best_idx = -1
        best_score = None

        for i, c in enumerate(candidates):
            if c.get("_picked"):
                continue
            cid = str(c["chunk_id"])
            pos = id_to_pos.get(cid)
            if pos is None:
                continue
            v = emb[pos]  # emb уже L2-нормализован
            rel = float(c.get("similarity", 0.0))

            if not selected_vecs:
                mmr_score = rel
            else:
                max_red = max(float(np.dot(v, sv)) for sv in selected_vecs)
                mmr_score = lambda_mult * rel - (1.0 - lambda_mult) * max_red

            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx < 0:
            break

        candidates[best_idx]["_picked"] = True
        candidates[best_idx]["mmr_score"] = float(best_score) if best_score is not None else 0.0
        selected.append(candidates[best_idx])

        cid = str(candidates[best_idx]["chunk_id"])
        pos = id_to_pos.get(cid)
        if pos is not None:
            selected_vecs.append(emb[pos])

    for c in candidates:
        if "_picked" in c:
            del c["_picked"]

    return selected


def load_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(meta_path):
        meta[row["chunk_id"]] = row
    return meta


def build_prompt(query: str, quotes: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Возвращает (system, user_prompt).

    Режимы:
      - EXTRACT: в контексте есть явное правило/факт, который отвечает на вопрос
      - INFER: прямой формулировки нет, но из контекста следует ответ (обобщение, частный случай, исключение)
      - NOT_FOUND: только если контекст НЕ содержит ни правила, ни определения, ни ограничений,
                   ни исключений, ни процедур, которые относятся к теме вопроса
    """
    system = (
        "Ты корпоративный ассистент по нормативным документам. "
        "Ты ОБЯЗАН отвечать строго на основе контекста [Q..] и не добавлять факты извне. "
        "Если в контексте есть хоть какая-то релевантная норма/ограничение/исключение/определение по теме вопроса, "
        "ты НЕ ИМЕЕШЬ ПРАВА выбирать NOT_FOUND: выбери EXTRACT или INFER. "
        "NOT_FOUND разрешён только когда контекст по теме вопроса пуст (нет релевантных норм/определений/ограничений). "
        "Если выбираешь NOT_FOUND, ты обязан кратко объяснить, чего именно не хватает в контексте."
    )

    blocks: List[str] = []
    for i, q in enumerate(quotes, start=1):
        src = q.get("source", {}) or {}
        blocks.append(
            f"[Q{i}] SOURCE: doc_id={src.get('doc_id')}; chunk_id={src.get('chunk_id')}; "
            f"pages={src.get('pages')}; paras={src.get('paras')}\n"
            f"CONTEXT:\n{q.get('quote','')}"
        )

    context = "\n\n".join(blocks) if blocks else "(пусто)"

    user = f"""ВОПРОС:
{query}

КОНТЕКСТ [Q..] (единственный источник истины):
{context}

ЗАДАНИЕ:
1) EXTRACT: если в контексте есть явная формулировка правила/факта/исключения, которая отвечает на вопрос.
2) INFER: если прямой формулировки нет, но из контекста следует ответ (например, есть исключение, определение,
   ограничение, процедура, или частный случай), сформулируй вывод и отметь "ВЫВОД".
3) NOT_FOUND: только если контекст (Q..) по теме вопроса пуст. Если контекст содержит хотя бы частично релевантную норму,
   определение, ограничение, исключение или процедуру, NOT_FOUND запрещён.

ФОРМАТ (строго):
- Режим: EXTRACT|INFER|NOT_FOUND
- Ответ: ...
- Основание: [Q1], [Q2]...
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


def retrieve_within_doc(
    ids: List[str],
    emb: np.ndarray,
    chunk_doc: Dict[str, Optional[str]],
    qvec: List[float],
    target_doc_id: str,
) -> List[Tuple[str, float]]:
    """Exact cosine-similarity search over ALL chunks of a single document.

    Semantics of --restrict_doc should be: search within the target document,
    not global-top then filter.
    """
    if not target_doc_id:
        return []

    doc_idx = [i for i, cid in enumerate(ids) if chunk_doc.get(cid) == target_doc_id]
    if not doc_idx:
        return []

    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)

    M = emb[doc_idx]
    scores = (M @ q.T).reshape(-1)

    order = np.argsort(-scores)
    out: List[Tuple[str, float]] = []
    for j in order:
        jj = int(j)
        out.append((ids[doc_idx[jj]], float(scores[jj])))
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
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid reranking with BM25 over the candidate set",
    )
    ap.add_argument(
        "--bm25_alpha",
        type=float,
        default=0.65,
        help="Hybrid weight 0..1 (1=only BM25, 0=only similarity)",
    )
    ap.add_argument(
    "--rerank_pool_multiplier",
    type=int,
    default=5,
    help="Сколько кандидатов держать до rerank (множитель от top_k).",
    )
    ap.add_argument(
        "--rerank_pool_min",
        type=int,
        default=100,
        help="Минимальный размер пула кандидатов до rerank (если возможно).",
    )
    ap.add_argument(
        "--mmr",
        action="store_true",
        help="Включить MMR-диверсификацию при выборе контекста для LLM.",
    )
    ap.add_argument(
        "--mmr_lambda",
        type=float,
        default=0.7,
        help="MMR lambda в [0..1]: больше -> релевантность, меньше -> разнообразие.",
    )
    ap.add_argument(
        "--log_candidates",
        type=int,
        default=0,
        help="Если >0, сохранять top-N кандидатов до/после rerank в results для дебага.",
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
    logger.info(f"hybrid={bool(args.hybrid)} bm25_alpha={float(args.bm25_alpha)}")

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
                "hybrid": bool(args.hybrid),
                "bm25_alpha": float(args.bm25_alpha)
            
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
    # Load embeddings aligned with chunk_ids.txt (needed for exact within-doc search)
    id_to_pos = {cid: i for i, cid in enumerate(ids)}
    emb_path = os.path.join(index_dir, "embeddings.npy")
    emb = np.load(emb_path).astype(np.float32)
    faiss.normalize_L2(emb)

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

            # Retrieval
            if restrict_doc and target_doc_id:
                # True within-document search: score ALL chunks of the target document
                retrieved = retrieve_within_doc(
                    ids=ids, emb=emb, chunk_doc=chunk_doc, qvec=qvec, target_doc_id=target_doc_id
                )
            else:
                # Global search over the whole corpus
                retrieved = retrieve_candidates(index, ids, qvec, k_search)

            # Optional hybrid rerank over the candidate set
            # Build candidate pool BEFORE rerank (so rerank has room to work)
            pool_size = max(int(args.rerank_pool_min), int(args.rerank_pool_multiplier) * int(top_k))
            retrieved_pool = retrieved[:pool_size] if len(retrieved) > pool_size else retrieved

            # Prepare verbose candidate list
            candidates_pre: List[Dict[str, Any]] = []
            for ch_id, sim in retrieved_pool:
                m = meta.get(ch_id, {}) or {}
                pages, paras = format_where(m)
                candidates_pre.append(
                    {
                        "chunk_id": ch_id,
                        "similarity": float(sim),
                        "doc_id": m.get("doc_id"),
                        "pages": pages,
                        "paras": paras,
                    }
                )

            # Hybrid rerank (BM25 + similarity)
            candidates_post: List[Dict[str, Any]] = candidates_pre
            if args.hybrid:
                # BM25 over pool
                docs_tokens = [bm25_tokenize(chunk_text_by_id.get(c["chunk_id"], "")) for c in candidates_pre]
                k1 = 1.5
                b = 0.75
                N = len(docs_tokens)
                df = Counter()
                doc_lens = []
                for toks in docs_tokens:
                    doc_lens.append(len(toks))
                    df.update(set(toks))
                avgdl = (sum(doc_lens) / max(1, N)) if N else 0.0

                def bm25_score(qtoks: List[str], idx: int) -> float:
                    score = 0.0
                    toks = docs_tokens[idx]
                    if not toks:
                        return 0.0
                    tf = Counter(toks)
                    dl = doc_lens[idx]
                    for term in qtoks:
                        if term not in tf:
                            continue
                        n_q = df.get(term, 0)
                        idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                        f = tf[term]
                        denom = f + k1 * (1.0 - b + b * (dl / (avgdl + 1e-9)))
                        score += idf * (f * (k1 + 1.0) / (denom + 1e-9))
                    return score

                q_toks = bm25_tokenize(query)
                bm25_scores = [bm25_score(q_toks, i) for i in range(N)]
                if bm25_scores:
                    mn, mx = min(bm25_scores), max(bm25_scores)
                    rng = (mx - mn) if (mx - mn) > 1e-9 else 1.0
                    bm25_norm = [(s - mn) / rng for s in bm25_scores]
                else:
                    bm25_norm = [0.0 for _ in range(N)]

                alpha = float(args.bm25_alpha)
                tmp: List[Dict[str, Any]] = []
                for i, c in enumerate(candidates_pre):
                    sim = float(c.get("similarity", 0.0))
                    b25 = float(bm25_norm[i])
                    hybrid = alpha * b25 + (1.0 - alpha) * sim
                    cc = dict(c)
                    cc["bm25"] = b25
                    cc["hybrid_score"] = hybrid
                    tmp.append(cc)

                tmp.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
                candidates_post = tmp
            else:
                candidates_post = sorted(candidates_pre, key=lambda x: x.get("similarity", 0.0), reverse=True)

            # keep top_k after rerank
            candidates_post = candidates_post[: int(top_k)]

            # MMR diversification for contexts sent to LLM
            if args.mmr:
                selected_for_llm = mmr_select(
                    candidates=candidates_post,
                    qvec=qvec,
                    emb=emb,
                    id_to_pos=id_to_pos,
                    max_select=max(1, int(args.max_quotes)),
                    lambda_mult=float(args.mmr_lambda),
                )
            else:
                selected_for_llm = candidates_post[: max(1, int(args.max_quotes))]

            # store debug candidates if requested
            if int(args.log_candidates) > 0:
                Nlog = int(args.log_candidates)
                case_payload["candidates_pre_rerank"] = candidates_pre[:Nlog]
                case_payload["candidates_post_rerank"] = candidates_post[:Nlog]
                case_payload["selected_for_llm"] = selected_for_llm

            # Backward compatible 'retrieved' list (post-rerank)
            retrieved = [(c["chunk_id"], float(c.get("similarity", 0.0))) for c in candidates_post]


            if restrict_doc and target_doc_id:
                doc_total = sum(1 for _cid in ids if chunk_doc.get(_cid) == target_doc_id)
                case_payload["notes"].append(f"within_doc_total_chunks={doc_total} retrieved_len={len(retrieved)}")

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
            if not retrieved:
                case_payload["answer"] = "NOT_FOUND"
                case_payload["status"] = "NOT_FOUND_EMPTY_RETRIEVAL"
                out_rows.append(case_payload)
                save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
                continue

            # Similarity threshold is useful for global search, but for strict per-document runs
            # it can hide relevant chunks simply because the doc is small/narrow.
            if (not restrict_doc) and retrieved[0][1] < min_similarity:
                case_payload["answer"] = "NOT_FOUND"
                case_payload["status"] = "NOT_FOUND_BELOW_SIMILARITY_THRESHOLD"
                out_rows.append(case_payload)
                save_case_markdown(os.path.join(md_dir, f"{cid}.md"), case_payload)
                continue

            # Extract quotes
            quotes: List[Dict[str, Any]] = []
            for c in selected_for_llm:
                ch_id = c["chunk_id"]
                sim = float(c.get("similarity", 0.0))
                full_text = chunk_text_by_id.get(ch_id, "") or ""

                qc = extract_best_quote(
                    chunk_text=full_text,
                    query=query,
                    max_quote_chars=max_quote_chars,
                    window_sentences=2,
                )

                quote_text = (qc.text or "").strip() if qc else ""
                quote_score = float(getattr(qc, "score", 0.0)) if qc else 0.0
                hit_words = list(getattr(qc, "hit_words", [])) if qc else []

                # fallback: если extractor не нашёл цитату, всё равно даём кусок чанка
                if not quote_text:
                    quote_text = full_text.strip()[: int(max_quote_chars)]
                    quote_score = 0.0
                    hit_words = []

                if not quote_text:
                    continue

                m = meta.get(ch_id, {}) or {}
                pages, paras = format_where(m)

                quotes.append(
                    {
                        "chunk_id": ch_id,
                        "similarity": float(sim),
                        "quote_score": float(quote_score),
                        "hit_words": hit_words,
                        "quote": quote_text,
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
                case_payload["status"] = "NOT_FOUND_NO_CONTEXT"
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

            # Second-try logic: if model returned NOT_FOUND despite having context,
            # forbid NOT_FOUND and ask for EXTRACT/INFER using the same quotes.
            is_nf = False
            a0 = normalize_text(answer)
            if a0 == "NOT_FOUND":
                is_nf = True
            else:
                for _ln in (answer or "").splitlines():
                    if "Режим:" in _ln and "NOT_FOUND" in _ln:
                        is_nf = True
                        break

            if is_nf and quotes:
                system2 = (
                    system
                    + " ВАЖНО: режим NOT_FOUND запрещён. Выбери EXTRACT или INFER. "
                      "Отвечай строго по [Q..]. Если данных недостаточно для точного ответа, "
                      "выбери INFER и явно укажи ограничения/что именно отсутствует."
                )
                answer2 = client.generate(
                    model=llm_model,
                    prompt=prompt,
                    system=system2,
                    temperature=0.1,
                ).strip()

                a2 = normalize_text(answer2)
                is_nf2 = (a2 == "NOT_FOUND") or any(
                    ("Режим:" in _ln and "NOT_FOUND" in _ln) for _ln in (answer2 or "").splitlines()
                )

                if not is_nf2:
                    case_payload["notes"].append("second_try_triggered: replaced NOT_FOUND with EXTRACT/INFER")
                    answer = answer2
                else:
                    case_payload["notes"].append("second_try_triggered: still NOT_FOUND")

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
