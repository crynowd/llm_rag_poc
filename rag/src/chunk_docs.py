import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils import ensure_dir, load_json, load_jsonl, make_run_dir, setup_logger, write_jsonl
from ollama_client import OllamaClient

import re

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

import json

def llm_split_into_segments(client, model: str, page_text: str, target_chars: int = 1200) -> list[str]:
    system = (
        "Ты инструмент для разметки текста. "
        "НЕЛЬЗЯ переписывать, исправлять, сокращать или добавлять текст. "
        "Нужно ТОЛЬКО разбить исходный текст на смысловые сегменты."
    )

    prompt = f"""
Разбей текст на сегменты (примерно {target_chars}–{target_chars*2} символов каждый), сохраняя исходный текст ПОСИМВОЛЬНО.
Верни строго JSON вида:
{{"segments": ["...", "...", ...]}}

Текст:
<<<
{page_text}
>>>
"""

    raw = client.generate(model=model, prompt=prompt, system=system, temperature=0.0).strip()

    # вытащим JSON (на случай если модель добавила мусор вокруг)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        obj = json.loads(raw[start:end+1])
    except Exception:
        return []

    segs = obj.get("segments")
    if not isinstance(segs, list):
        return []

    # фильтруем: сегмент должен быть подстрокой исходного текста
    ok = []
    for s in segs:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if len(s) < 50:
            continue
        if s in page_text:
            ok.append(s)

    return ok

def flatten_segments_for_chunking(
    rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    client: Optional["OllamaClient"] = None,
) -> List[Dict[str, Any]]:
    """
    Приводит сегменты в единый список 'атомов' для чанкинга:
    - PDF: страница -> список сегментов (heuristic или LLM)
    - DOCX: абзац/строка таблицы как есть
    """
    chunking_cfg = cfg.get("chunking", {}) if isinstance(cfg, dict) else {}
    pdf_chunker = chunking_cfg.get("pdf_chunker", "heuristic")  # "heuristic" | "llm"
    target_chars = int(chunking_cfg.get("pdf_llm_target_chars", 1200))

    llm_model = (cfg.get("ollama", {}) or {}).get("llm_model")

    out: List[Dict[str, Any]] = []
    for r in rows:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue

        if r.get("doc_type") == "pdf":
            cleaned = cleanup_pdf_text(txt)

            # --- LLM-aware chunking (если включено и есть client/model) ---
            if pdf_chunker == "llm" and client is not None and llm_model:
                segs = llm_split_into_segments(
                    client=client,
                    model=llm_model,
                    page_text=cleaned,
                    target_chars=target_chars,
                )
                if segs:
                    for i, seg in enumerate(segs, start=1):
                        if len(seg.strip()) < 20:
                            continue
                        out.append({**r, "text": seg, "para_idx": i})
                    continue  # важно: не падать в heuristic ниже

            # --- fallback: heuristic ---
            paras = split_into_paragraphs(cleaned)
            for i, para in enumerate(paras, start=1):
                if len(para.strip()) < 20:
                    continue
                out.append({**r, "text": para, "para_idx": i})

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
    max_chars = int(cfg["chunking"]["max_chars"])
    overlap_chars = int(cfg["chunking"]["overlap_chars"])

    logger.info(f"chunking.max_chars={max_chars}")
    logger.info(f"chunking.overlap_chars={overlap_chars}")

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

        pdf_chunker = cfg.get("chunking", {}).get("pdf_chunker", "heuristic")
        client = OllamaClient() if pdf_chunker == "llm" else None

        atoms = flatten_segments_for_chunking(rows, cfg=cfg, client=client)

        logger.info(f"doc_atoms doc_id={doc_id} atoms={len(atoms)}")

        doc_chunks = chunk_atoms(atoms, max_chars=max_chars, overlap_chars=overlap_chars)
        all_chunks.extend(doc_chunks)

        lens = [c["char_len"] for c in doc_chunks]
        st = stats_charlens(lens)
        too_short = sum(1 for x in lens if x < 200)
        too_long = sum(1 for x in lens if x > max_chars + 200)  # с запасом на переносы/overlap

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
