#!/usr/bin/env python3
"""Run benchmark cases through existing RAG pipeline (retrieval + quotes + answer).
format_version: 1
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss  # type: ignore
import numpy as np


def _setup_imports() -> Path:
    bench_dir = Path(__file__).resolve().parent
    project_dir = bench_dir.parent  # rag/
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    return project_dir


PROJECT_DIR = _setup_imports()

from src.answer import build_prompt, load_meta  # noqa: E402
from src.ollama_client import OllamaClient  # noqa: E402
from src.quote_extractor import extract_best_quote  # noqa: E402
from src.utils import load_json, load_jsonl  # noqa: E402


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"[{ts}] run_bench.py | {message}\n")


def read_cases(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_shuffle_and_limit(
    rows: List[Dict[str, Any]],
    shuffle_seed: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    out = list(rows)
    if shuffle_seed != 0:
        rnd = random.Random(shuffle_seed)
        rnd.shuffle(out)
    if limit is not None:
        out = out[: max(0, int(limit))]
    return out


def normalize_not_found(
    answer_text: str, is_not_found: bool, stop_reason: str
) -> tuple[str, bool, str]:
    text = answer_text or ""
    has_nf = "NOT_FOUND" in text
    if has_nf:
        is_not_found = True
    if is_not_found:
        if not text.startswith("- Ответ: NOT_FOUND"):
            if "\n- Основание:" in text:
                _, basis = text.split("\n- Основание:", 1)
                text = "- Ответ: NOT_FOUND\n- Основание:" + basis
            else:
                text = "- Ответ: NOT_FOUND"
        stop_reason = "not_found"
    return text, is_not_found, stop_reason


def run_case(
    case: Dict[str, Any],
    cfg: Dict[str, Any],
    index: faiss.Index,
    ids: List[str],
    meta: Dict[str, Dict[str, Any]],
    chunk_text_by_id: Dict[str, str],
    client: OllamaClient,
    max_quotes: int = 3,
) -> Dict[str, Any]:
    query = case["query"]
    embed_model = cfg["ollama"]["embedding_model"]
    llm_model = cfg["ollama"]["llm_model"]
    top_k = int(cfg["retrieval"]["top_k"])
    min_similarity = float(cfg["answering"]["min_similarity"])
    max_quote_chars = int(cfg["answering"]["max_quote_chars"])

    qvec = client.embed_one(embed_model, query)
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)

    retrieved: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
        if idx < 0 or idx >= len(ids):
            continue
        chunk_id = ids[idx]
        m = meta.get(chunk_id, {})
        pages = (
            f"{m.get('page_min')}-{m.get('page_max')}"
            if m.get("page_min") is not None
            else None
        )
        paras = (
            f"{m.get('para_min')}-{m.get('para_max')}"
            if m.get("para_min") is not None
            else None
        )
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "similarity": float(score),
                "doc_id": m.get("doc_id"),
                "pages": pages,
                "paras": paras,
            }
        )

    if not retrieved or retrieved[0]["similarity"] < min_similarity:
        answer_text, is_nf, stop_reason = normalize_not_found(
            "NOT_FOUND", True, "low_retrieval"
        )
        return {
            "answer_text": answer_text,
            "is_not_found": is_nf,
            "stop_reason": stop_reason,
            "retrieved": retrieved,
            "quotes": [],
            "error": None,
        }

    quotes: List[Dict[str, Any]] = []
    for r in retrieved:
        chunk_id = r["chunk_id"]
        full_text = chunk_text_by_id.get(chunk_id, "")
        qc = extract_best_quote(
            chunk_text=full_text,
            query=query,
            max_quote_chars=max_quote_chars,
            window_sentences=2,
        )
        if not qc.text:
            continue
        quotes.append(
            {
                "chunk_id": chunk_id,
                "similarity": r["similarity"],
                "quote_score": qc.score,
                "hit_words": qc.hit_words,
                "quote": qc.text,
                "source": {
                    "doc_id": r["doc_id"],
                    "chunk_id": chunk_id,
                    "pages": r["pages"],
                    "paras": r["paras"],
                },
            }
        )

    if not quotes:
        answer_text, is_nf, stop_reason = normalize_not_found(
            "NOT_FOUND", True, "no_quote"
        )
        return {
            "answer_text": answer_text,
            "is_not_found": is_nf,
            "stop_reason": stop_reason,
            "retrieved": retrieved,
            "quotes": [],
            "error": None,
        }

    quotes.sort(key=lambda x: (x["similarity"], x["quote_score"]), reverse=True)
    quotes = quotes[: max(1, int(max_quotes))]
    for i, qd in enumerate(quotes, start=1):
        qd["q_id"] = f"Q{i}"

    system, prompt = build_prompt(query, quotes)
    answer = client.generate(
        model=llm_model,
        prompt=prompt,
        system=system,
        temperature=0.1,
    ).strip()

    answer, is_nf, stop_reason = normalize_not_found(
        answer, False, "answered"
    )
    return {
        "answer_text": answer,
        "is_not_found": is_nf,
        "stop_reason": stop_reason,
        "retrieved": retrieved,
        "quotes": quotes,
        "error": None,
    }


def _fmt_hhmmss(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark through existing RAG pipeline")
    ap.add_argument("--cases", required=True, help="Path to normalized cases JSONL")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=None, help="Optional number of cases")
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="0 means keep order; non-zero means deterministic shuffle",
    )
    ap.add_argument(
        "--progress",
        type=str,
        default="true",
        help="Show progress in terminal (true/false). Default: true",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="Update progress not more often than every N cases. Default: 1",
    )
    ap.add_argument(
        "--debug_traceback",
        action="store_true",
        help="Write full traceback to log on errors",
    )
    args = ap.parse_args()

    cases_path = Path(args.cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "bench_run_result.jsonl"
    log_path = out_dir / "log.txt"

    show_progress = str(args.progress).strip().lower() not in {"false", "0", "no"}
    progress_every = max(1, int(args.progress_every))

    append_log(log_path, "START run")
    append_log(
        log_path,
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()} "
        f"cases={cases_path} out_jsonl={out_jsonl} limit={args.limit} seed={args.shuffle_seed}",
    )

    cfg = load_json(str(PROJECT_DIR / "config.json"))
    index_dir = PROJECT_DIR / "data" / "index"
    chunks_path = PROJECT_DIR / "data" / "chunks" / "chunks.jsonl"
    meta_path = index_dir / "chunk_meta.jsonl"
    ids_path = index_dir / "chunk_ids.txt"
    faiss_path = index_dir / "faiss.index"

    chunks = load_jsonl(str(chunks_path))
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    meta = load_meta(str(meta_path))
    with ids_path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    index = faiss.read_index(str(faiss_path))

    cases = read_cases(cases_path)
    cases = maybe_shuffle_and_limit(cases, args.shuffle_seed, args.limit)
    total_cases = len(cases)
    append_log(log_path, f"cases_to_run={total_cases}")

    client = OllamaClient()
    success = 0
    failed = 0
    not_found_count = 0
    start_ts = time.monotonic()

    with out_jsonl.open("w", encoding="utf-8", newline="\n") as out_f:
        for i, case in enumerate(cases, start=1):
            case_id = case.get("case_id", f"row_{i}")
            case_start = time.monotonic()
            try:
                result = run_case(
                    case=case,
                    cfg=cfg,
                    index=index,
                    ids=ids,
                    meta=meta,
                    chunk_text_by_id=chunk_text_by_id,
                    client=client,
                )
                timing_ms = int((time.monotonic() - case_start) * 1000)
                status = "OK"
                err = None
                row = {
                    "format_version": 1,
                    "case_id": case_id,
                    "source_case_id": case.get("source_case_id"),
                    "doc_id": case.get("doc_id"),
                    "query": case.get("query"),
                    "expected": case.get("expected"),
                    "run": result,
                    "timing_ms": timing_ms,
                    "status": status,
                    "error": err,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                out_f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                success += 1
                if result.get("is_not_found"):
                    not_found_count += 1
                append_log(
                    log_path,
                    f"case_id={case_id} status=OK duration_ms={timing_ms}",
                )
            except Exception as exc:
                failed += 1
                timing_ms = int((time.monotonic() - case_start) * 1000)
                status = "FAIL"
                err = {"type": type(exc).__name__, "message": str(exc)}
                err_row = {
                    "format_version": 1,
                    "case_id": case_id,
                    "source_case_id": case.get("source_case_id"),
                    "doc_id": case.get("doc_id"),
                    "query": case.get("query"),
                    "expected": case.get("expected"),
                    "run": {
                        "answer_text": "",
                        "is_not_found": False,
                        "stop_reason": "error",
                        "retrieved": [],
                        "quotes": [],
                        "error": None,
                    },
                    "timing_ms": timing_ms,
                    "status": status,
                    "error": err,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                out_f.write(json.dumps(err_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                append_log(
                    log_path,
                    f"case_id={case_id} status=FAIL duration_ms={timing_ms} "
                    f"error_type={type(exc).__name__} error_message={exc}",
                )
                if args.debug_traceback:
                    append_log(log_path, "traceback:")
                    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    for line in tb.rstrip().splitlines():
                        append_log(log_path, line)

            if show_progress and (i % progress_every == 0 or i == total_cases):
                elapsed = int(time.monotonic() - start_ts)
                avg = elapsed / max(i, 1)
                remaining = int(avg * max(total_cases - i, 0))
                pct = (i / max(total_cases, 1)) * 100.0
                prog = (
                    f"[ {i:>3}/{total_cases:<3} | {pct:5.1f}% ] "
                    f"elapsed={_fmt_hhmmss(elapsed)} eta={_fmt_hhmmss(remaining)} case_id={case_id}"
                )
                print("\r" + prog, end="", flush=True)

    append_log(log_path, f"done success={success} failed={failed} not_found={not_found_count}")
    if show_progress:
        print()
    print(f"total={total_cases} ok={success} failed={failed} not_found={not_found_count}")
    print(f"written: {out_jsonl}")
    print(f"log: {log_path}")
    print(f"cases_total: {total_cases}")
    print(f"cases_success: {success}")
    print(f"cases_failed: {failed}")


if __name__ == "__main__":
    main()
