import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils import ensure_dir, load_json, load_jsonl, make_run_dir, setup_logger, write_jsonl

import re

# --- PDF sentence-based chunking helpers ---

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> List[str]:
    """Naive sentence splitter for PDF text.

    PDF is messy (numbering, abbreviations). We'll add a guardrail later.
    """
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    sents = _SENT_SPLIT_RE.split(text)
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if len(s) >= 2:
            out.append(s)
    return out


def pack_sentences(sentences: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """Pack sentences into chunks up to max_chars.

    Overlap is implemented via *carry-over sentences* (no index backtracking)
    to avoid infinite loops.
    """
    if not sentences:
        return []

    chunks: List[str] = []
    i = 0
    n = len(sentences)
    carry: List[str] = []

    while i < n:
        start_i = i

        buf: List[str] = list(carry)
        buf_len = len(" ".join(buf)) if buf else 0

        while i < n:
            s = (sentences[i] or "").strip()
            if not s:
                i += 1
                continue

            # If chunk is empty and sentence itself is too long, keep it whole.
            if not buf and len(s) > max_chars:
                buf = [s]
                buf_len = len(s)
                i += 1
                break

            add_len = len(s) + (1 if buf else 0)
            if buf_len + add_len <= max_chars:
                buf.append(s)
                buf_len += add_len
                i += 1
                continue
            break

        # Safety: ensure forward progress
        if i == start_i and i < n:
            s = (sentences[i] or "").strip()
            if s:
                buf = [s]
            i += 1

        chunk_text = " ".join([x for x in buf if x]).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if overlap_chars and overlap_chars > 0:
            carry = []
            carry_len = 0
            for s in reversed(buf):
                if not s:
                    continue
                carry.append(s)
                carry_len += len(s) + 1
                if carry_len >= overlap_chars:
                    break
            carry = list(reversed(carry))
        else:
            carry = []

    return chunks

def cleanup_pdf_text(t: str) -> str:
    t = (t or "").replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)

    lines = [ln.strip() for ln in t.split("\n")]
    out = []
    for ln in lines:
        if not ln:
            out.append("")
            continue
        if not out:
            out.append(ln)
            continue

        prev = out[-1]
        # если перенос внутри предложения (предыдущая строка не закончена) — склеиваем
        if prev and not prev.endswith((".", "!", "?", ":", ";")) and re.match(r"^[а-яa-z]", ln):
            out[-1] = prev + " " + ln
        else:
            out.append(ln)

    t = "\n".join(out)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def split_into_paragraphs(t: str) -> list[str]:
    # Сначала режем по пустым строкам
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    return paras


def split_pdf_text_to_blocks(text: str) -> List[str]:
    # Простое разбиение "как абзацы": двойной перенос строки.
    # Детерминированно, без LLM, для PoC достаточно.
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


def flatten_segments_for_chunking(
    rows: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    logger,
) -> List[Dict[str, Any]]:
    """
    Приводит сегменты в единый список 'атомов' для чанкинга:
    - PDF: страница -> список абзацев (blocks)
    - DOCX: абзац/строка таблицы как есть
    """
    out: List[Dict[str, Any]] = []

    ch_cfg = cfg.get("chunking", {})
    pdf_max_chars = int(ch_cfg.get("pdf_max_chars", 1400))
    pdf_overlap_chars = int(ch_cfg.get("pdf_overlap_chars", 200))

    pdf_total = sum(1 for x in rows if x.get("doc_type") == "pdf")
    pdf_seen = 0

    for r in rows:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue

        if r.get("doc_type") == "pdf":
            pdf_seen += 1
            if pdf_total and (pdf_seen % 10 == 0 or pdf_seen == pdf_total):
                logger.info(f"pdf_progress doc_id={r.get('doc_id')} page={pdf_seen}/{pdf_total}")

            cleaned = cleanup_pdf_text(txt)
            sents = split_into_sentences(cleaned)

            # Guardrail: sentence splitter may explode on numbering/abbrev.
            if len(sents) > 5000:
                logger.info(
                    f"too_many_sentences doc_id={r.get('doc_id')} page={r.get('page')} sentences={len(sents)} "
                    f"text_len={len(cleaned)} -> fallback paragraphs"
                )
                sents = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

            pieces = pack_sentences(sents, max_chars=pdf_max_chars, overlap_chars=pdf_overlap_chars)
            for i, piece in enumerate(pieces, start=1):
                if len(piece.strip()) < 20:
                    continue
                out.append({**r, "text": piece, "para_idx": i})
        else:
            out.append(r)
    return out


def make_chunk_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}::{idx:05d}"


def chunk_atoms(
    atoms: List[Dict[str, Any]],
    max_chars: int,
    overlap_chars: int,
) -> List[Dict[str, Any]]:
    """
    Детерминированный чанкинг:
    - собираем атомы последовательно до max_chars
    - при переполнении закрываем чанк
    - overlap делаем текстовый + сохраняем последний атом как "якорь происхождения"
    """
    chunks: List[Dict[str, Any]] = []

    buf_text: str = ""
    buf_atoms: List[Dict[str, Any]] = []
    idx = 0

    def flush():
        nonlocal buf_text, buf_atoms, idx
        if not buf_text.strip():
            buf_text = ""
            buf_atoms = []
            return

        doc_id = buf_atoms[0]["doc_id"]
        doc_type = buf_atoms[0]["doc_type"]
        source_path = buf_atoms[0]["source_path"]

        pages = [a["page"] for a in buf_atoms if a.get("page") is not None]
        paras = [a["para_idx"] for a in buf_atoms if a.get("para_idx") is not None]

        chunk = {
            "chunk_id": make_chunk_id(doc_id, idx),
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_path": source_path,
            "page_min": min(pages) if pages else None,
            "page_max": max(pages) if pages else None,
            "para_min": min(paras) if paras else None,
            "para_max": max(paras) if paras else None,
            "char_len": len(buf_text),
            "text": buf_text.strip(),
        }
        chunks.append(chunk)
        idx += 1

        if overlap_chars > 0:
            # Текстовый overlap + последний атом как якорь для источника
            buf_text = buf_text[-overlap_chars:]
            buf_atoms = buf_atoms[-1:]
        else:
            buf_text = ""
            buf_atoms = []

    for a in atoms:
        t = (a.get("text") or "").strip()
        if not t:
            continue

        candidate = buf_text + ("\n" if buf_text else "") + t
        if buf_text and len(candidate) > max_chars:
            flush()

        # после flush буфер мог стать overlap'ом, добавляем текст
        buf_text = buf_text + ("\n" if buf_text else "") + t
        buf_atoms.append(a)

        # если один атом сам по себе больше max_chars, мы его всё равно положим как отдельный чанк
        if len(buf_text) > max_chars and len(buf_atoms) == 1:
            flush()

    flush()
    return chunks


def stats_charlens(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": None, "avg": None, "max": None}
    return {
        "min": min(values),
        "avg": sum(values) // len(values),
        "max": max(values),
    }


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    run_dir = make_run_dir(project_dir)
    log_path = os.path.join(run_dir, "chunk.log")
    logger = setup_logger("rag_chunking", log_path)

    logger.info("=== CHUNKING START ===")

    cfg = load_json(os.path.join(project_dir, "config.json"))
    ch_cfg = cfg.get("chunking", {})

    # global defaults
    max_chars_default = int(ch_cfg.get("max_chars", 3500))
    overlap_default = int(ch_cfg.get("overlap_chars", 250))

    logger.info(f"chunking.max_chars={max_chars_default}")
    logger.info(f"chunking.overlap_chars={overlap_default}")

    parsed_path = os.path.join(project_dir, "data", "parsed", "parsed.jsonl")
    parsed = load_jsonl(parsed_path)
    logger.info(f"parsed_segments={len(parsed)}")

    # group by doc_id
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for r in parsed:
        by_doc.setdefault(r["doc_id"], []).append(r)

    all_chunks: List[Dict[str, Any]] = []

    for doc_id, rows in tqdm(by_doc.items(), desc="Chunking documents"):
        doc_type = rows[0].get("doc_type")
        logger.info(f"doc_start doc_id={doc_id} doc_type={doc_type} segments={len(rows)}")

        # per-doc params
        if doc_type == "pdf":
            max_chars = int(ch_cfg.get("pdf_max_chars", max_chars_default))
            # overlap for PDFs is handled inside sentence packing -> avoid double overlap here
            overlap = 0
        elif doc_type == "docx":
            max_chars = int(ch_cfg.get("docx_max_chars", max_chars_default))
            overlap = int(ch_cfg.get("docx_overlap_chars", overlap_default))
        else:
            max_chars = max_chars_default
            overlap = overlap_default

        logger.info(f"chunk_params doc_id={doc_id} doc_type={doc_type} max_chars={max_chars} overlap={overlap}")

        atoms = flatten_segments_for_chunking(rows, cfg=cfg, logger=logger)
        logger.info(f"doc_atoms doc_id={doc_id} atoms={len(atoms)}")

        doc_chunks = chunk_atoms(atoms, max_chars=max_chars, overlap_chars=overlap)
        all_chunks.extend(doc_chunks)

        lens = [c["char_len"] for c in doc_chunks]
        st = stats_charlens(lens)
        too_short = sum(1 for x in lens if x < 200)
        too_long = sum(1 for x in lens if x > max_chars + 200)  # с запасом на переносы

        logger.info(
            f"doc_done doc_id={doc_id} chunks={len(doc_chunks)} "
            f"min_chars={st['min']} avg_chars={st['avg']} max_chars={st['max']} "
            f"too_short(<200)={too_short} too_long(>{max_chars+200})={too_long}"
        )

    out_dir = os.path.join(project_dir, "data", "chunks")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "chunks.jsonl")
    write_jsonl(out_path, all_chunks)

    lens_all = [c["char_len"] for c in all_chunks]
    st_all = stats_charlens(lens_all)

    logger.info(f"[OK] Written: {out_path}")
    logger.info(
        f"total_chunks={len(all_chunks)} min_chars={st_all['min']} avg_chars={st_all['avg']} max_chars={st_all['max']}"
    )
    logger.info("=== CHUNKING END ===")

    print(f"[OK] Written: {out_path}")
    print(f"[STATS] total_chunks={len(all_chunks)}")
    print(f"[LOG] {log_path}")


if __name__ == "__main__":
    main()
