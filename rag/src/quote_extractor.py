import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Simplified Russian stopwords for PoC.
RU_STOPWORDS = {
    "и", "в", "во", "на", "по", "к", "ко", "о", "об", "обо", "от", "до", "из", "у",
    "за", "для", "при", "без", "над", "под", "про", "через", "между",
    "а", "но", "или", "ли", "же", "то", "это", "этот", "эта", "эти", "этом",
    "как", "какие", "какой", "какая", "какое", "каким", "какими",
    "что", "чтобы", "которые", "который", "которых",
    "все", "всех", "вся", "всё", "всего",
    "может", "могут", "можно", "должен", "должна", "должны",
    "включены", "включаться", "включен", "включает", "включают",
}

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)

# Soft penalty for frequent legal/technical tokens.
NOISE_TOKENS = {
    "договор",
    "срок",
    "дата",
    "настоящими",
    "правила",
    "доверительного",
    "управления",
}


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
    # unique in order
    seen = set()
    uniq = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
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


def sentence_score(
    sentence: str,
    terms: List[str],
    term_freq: Dict[str, int],
    noise_terms: List[str],
) -> Tuple[float, List[str], int]:
    s_norm = normalize_word(sentence)
    hits = [t for t in terms if t in s_norm]
    if not hits:
        return 0.0, [], 0

    uniq_hits = set(hits)
    coverage = len(uniq_hits)

    # approx-IDF: rare terms in chunk weigh more
    score = 0.0
    for t in uniq_hits:
        freq = term_freq.get(t, 1)
        score += 1.0 / (1.0 + float(freq))

    # soft penalty for frequent legal tokens not present in query terms
    penalty = 0.0
    for nt in noise_terms:
        if nt in s_norm:
            penalty += 0.1

    score = score + (0.25 * coverage) - penalty
    return score, hits, coverage


def extract_best_quote(
    chunk_text: str,
    query: str,
    max_quote_chars: int = 800,
    window_sentences: int = 2,
) -> QuoteCandidate:
    """
    Find best sentence by query terms and return windowed snippet.
    Deterministic, no external deps.
    """
    terms = extract_query_terms(query)
    term_set = set(terms)
    noise_terms = [t for t in NOISE_TOKENS if t not in term_set]

    # term frequencies within chunk for approx-IDF
    chunk_words = [normalize_word(w) for w in WORD_RE.findall(chunk_text)]
    term_freq: Dict[str, int] = {}
    for w in chunk_words:
        if w in term_set:
            term_freq[w] = term_freq.get(w, 0) + 1

    sents = split_sentences(chunk_text)
    best_idx = -1
    best = QuoteCandidate(text="", score=0.0, hit_words=[])
    best_cov = 0

    for i, s in enumerate(sents):
        sc, hits, cov = sentence_score(s, terms, term_freq, noise_terms)
        if sc > best.score or (sc == best.score and cov > best_cov):
            best = QuoteCandidate(text=s, score=sc, hit_words=hits)
            best_idx = i
            best_cov = cov

    if best_idx == -1:
        return QuoteCandidate(text="", score=0.0, hit_words=[])

    start = max(0, best_idx - (window_sentences - 1))
    end = min(len(sents), best_idx + window_sentences)
    snippet = " ".join(sents[start:end]).strip()

    if len(snippet) > max_quote_chars:
        snippet = snippet[:max_quote_chars].rstrip() + "..."

    return QuoteCandidate(text=snippet, score=best.score, hit_words=best.hit_words)
