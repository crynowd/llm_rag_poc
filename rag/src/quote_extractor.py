import re
from dataclasses import dataclass
from typing import List, Tuple


# Очень упрощённый список русских стоп-слов (хватает для PoC)
RU_STOPWORDS = {
    "и", "в", "во", "на", "по", "к", "ко", "о", "об", "обо", "от", "до", "из", "у",
    "за", "для", "при", "без", "над", "под", "про", "через", "между",
    "а", "но", "или", "ли", "же", "то", "это", "этот", "эта", "эти", "этом",
    "как", "какие", "какой", "какая", "какое", "каким", "какими",
    "что", "чтобы", "которые", "который", "которых",
    "все", "всех", "вся", "всё", "всего",
    "может", "могут", "можно", "должен", "должна", "должны",
    "включены", "включаться", "включен", "включает", "включают",  # спорно, но ок
}

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


@dataclass
class QuoteCandidate:
    text: str
    score: float
    hit_words: List[str]


def normalize_word(w: str) -> str:
    return w.lower().replace("ё", "е")


def extract_query_terms(query: str, min_len: int = 3) -> List[str]:
    words = [normalize_word(w) for w in WORD_RE.findall(query)]
    terms = []
    for w in words:
        if len(w) < min_len:
            continue
        if w in RU_STOPWORDS:
            continue
        terms.append(w)
    # уникализируем с сохранением порядка
    seen = set()
    uniq = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def split_sentences(text: str) -> List[str]:
    # Простое разбиение по . ! ? + переносы, без NLP.
    # Для нормативки обычно достаточно.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    # ещё режем длинные куски по ; (часто в перечнях)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 600:
            sub = [x.strip() for x in p.split(";") if x.strip()]
            out.extend(sub)
        else:
            out.append(p)
    return out


def sentence_score(sentence: str, terms: List[str]) -> Tuple[float, List[str]]:
    s_norm = normalize_word(sentence)
    hits = [t for t in terms if t in s_norm]
    if not hits:
        return 0.0, []
    # базовый скоринг: число попаданий + небольшой бонус за “плотность”
    # (чтобы не побеждали огромные предложения ни о чём)
    density = len(hits) / max(len(sentence), 1)
    score = float(len(hits)) + 50.0 * density
    return score, hits


def extract_best_quote(
    chunk_text: str,
    query: str,
    max_quote_chars: int = 800,
    window_sentences: int = 2,
) -> QuoteCandidate:
    """
    Ищем лучшие предложения по ключевым словам из запроса.
    Возвращаем кусок из N предложений (окно вокруг лучшего).
    Детерминированно.
    """
    terms = extract_query_terms(query)
    sents = split_sentences(chunk_text)

    best_idx = -1
    best = QuoteCandidate(text="", score=0.0, hit_words=[])

    for i, s in enumerate(sents):
        sc, hits = sentence_score(s, terms)
        if sc > best.score:
            best = QuoteCandidate(text=s, score=sc, hit_words=hits)
            best_idx = i

    if best_idx == -1:
        return QuoteCandidate(text="", score=0.0, hit_words=[])

    # окно вокруг лучшего предложения
    start = max(0, best_idx - (window_sentences - 1))
    end = min(len(sents), best_idx + window_sentences)
    snippet = " ".join(sents[start:end]).strip()

    # режем по лимиту
    if len(snippet) > max_quote_chars:
        snippet = snippet[:max_quote_chars].rstrip() + "..."

    return QuoteCandidate(text=snippet, score=best.score, hit_words=best.hit_words)
