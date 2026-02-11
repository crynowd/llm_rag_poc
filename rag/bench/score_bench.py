#!/usr/bin/env python3
"""Score benchmark run results and produce report.
format_version: 1
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

NBSP = "\u00a0"
NUM_RE = re.compile(r"\d[\d\s,\.\u00a0]*%?")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_numbers(text: str) -> str:
    if not text:
        return text
    text = text.replace(NBSP, " ")

    def _norm(m: re.Match) -> str:
        s = m.group(0)
        has_pct = s.endswith("%")
        if has_pct:
            s = s[:-1]
        s = s.replace(" ", "")
        s = s.replace(",", ".")
        if has_pct:
            s = s + "%"
        return s

    return NUM_RE.sub(_norm, text)


def normalize_for_match(text: str) -> str:
    return normalize_numbers(text).lower()


def extract_numbers(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace(NBSP, " ")
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group(0)
        has_pct = s.endswith("%")
        if has_pct:
            s = s[:-1]
        s = s.replace(" ", "").replace(",", ".")
        if has_pct:
            s = s + "%"
        out.append(s)
    return out


def has_citations(answer: str) -> bool:
    return "[Q" in answer


def must_contain_ok(answer: str, must_list: Iterable[str]) -> bool:
    a = normalize_for_match(answer)
    for m in must_list:
        if not m:
            continue
        if normalize_for_match(m) not in a:
            return False
    return True


def numbers_grounded(answer: str, quotes: Iterable[str]) -> bool:
    nums_a = extract_numbers(answer)
    if not nums_a:
        return True
    quotes_text = "\n".join(quotes)
    nums_q = set(extract_numbers(quotes_text))
    return all(n in nums_q for n in nums_a)


def detect_failure_reason(row: dict) -> str:
    expected = row.get("expected", {})
    exp_type = expected.get("type")
    run = row.get("run", {})
    answer = run.get("answer_text", "")
    is_nf = run.get("is_not_found", False)
    retrieved = run.get("retrieved", [])
    quotes = run.get("quotes", [])

    if exp_type == "not_found" and not is_nf:
        return "not_found_mismatch"
    if exp_type == "answer" and is_nf:
        return "not_found_mismatch"
    if not retrieved:
        return "retrieval_miss"
    if not quotes:
        return "quote_missing"
    if exp_type == "answer" and not has_citations(answer):
        return "citations_missing"
    if exp_type == "answer" and not must_contain_ok(answer, expected.get("must_contain", [])):
        return "must_contain_fail"
    if exp_type == "answer" and not numbers_grounded(answer, [q.get("quote", "") for q in quotes]):
        return "numbers_not_grounded"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Score benchmark run results")
    ap.add_argument("--cases", required=True, help="Path to normalized cases JSONL")
    ap.add_argument("--pred", required=True, help="Path to bench_run_result.jsonl")
    args = ap.parse_args()

    cases = {r["case_id"]: r for r in read_jsonl(Path(args.cases))}
    preds = read_jsonl(Path(args.pred))

    total = 0
    nf_total = 0
    nf_correct = 0
    cit_ok = 0
    must_ok = 0
    num_ok = 0
    ans_total = 0

    by_diff = defaultdict(lambda: Counter())
    by_tag = defaultdict(lambda: Counter())
    failures = []

    for row in preds:
        case_id = row.get("case_id")
        case = cases.get(case_id, {})
        expected = row.get("expected") or case.get("expected", {})
        exp_type = expected.get("type", "answer")
        run = row.get("run", {})
        answer = run.get("answer_text", "")
        quotes = [q.get("quote", "") for q in run.get("quotes", [])]

        total += 1
        if exp_type == "not_found":
            nf_total += 1
            if run.get("is_not_found", False):
                nf_correct += 1
        else:
            ans_total += 1
            if has_citations(answer):
                cit_ok += 1
            if must_contain_ok(answer, expected.get("must_contain", [])):
                must_ok += 1
            if numbers_grounded(answer, quotes):
                num_ok += 1

        reason = detect_failure_reason(row)
        if reason != "ok":
            failures.append(
                {
                    "case_id": case_id,
                    "reason": reason,
                    "query": row.get("query", ""),
                }
            )

        diff = str(case.get("difficulty", row.get("difficulty", "")))
        by_diff[diff]["total"] += 1
        if exp_type == "not_found":
            by_diff[diff]["nf_total"] += 1
            if run.get("is_not_found", False):
                by_diff[diff]["nf_correct"] += 1
        else:
            by_diff[diff]["ans_total"] += 1
            by_diff[diff]["cit_ok"] += 1 if has_citations(answer) else 0
            by_diff[diff]["must_ok"] += 1 if must_contain_ok(answer, expected.get("must_contain", [])) else 0
            by_diff[diff]["num_ok"] += 1 if numbers_grounded(answer, quotes) else 0

        tags = case.get("tags", row.get("tags", [])) or []
        for t in tags:
            by_tag[t]["total"] += 1
            if exp_type == "not_found":
                by_tag[t]["nf_total"] += 1
                if run.get("is_not_found", False):
                    by_tag[t]["nf_correct"] += 1
            else:
                by_tag[t]["ans_total"] += 1
                by_tag[t]["cit_ok"] += 1 if has_citations(answer) else 0
                by_tag[t]["must_ok"] += 1 if must_contain_ok(answer, expected.get("must_contain", [])) else 0
                by_tag[t]["num_ok"] += 1 if numbers_grounded(answer, quotes) else 0

    def rate(num: int, den: int) -> float:
        return 0.0 if den == 0 else (num / den)

    report = []
    report.append("# Bench Report")
    report.append("")
    report.append("## Summary")
    report.append(f"- total_cases: {total}")
    report.append(f"- not_found_accuracy: {rate(nf_correct, nf_total):.3f}")
    report.append(f"- citations_present_rate: {rate(cit_ok, ans_total):.3f}")
    report.append(f"- must_contain_pass_rate: {rate(must_ok, ans_total):.3f}")
    report.append(f"- numbers_grounded_rate: {rate(num_ok, ans_total):.3f}")
    report.append("")

    report.append("## Breakdown By Difficulty")
    report.append("| difficulty | total | not_found_accuracy | citations_present_rate | must_contain_pass_rate | numbers_grounded_rate |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for diff in sorted(by_diff.keys()):
        d = by_diff[diff]
        report.append(
            f"| {diff} | {d['total']} | {rate(d['nf_correct'], d['nf_total']):.3f} | "
            f"{rate(d['cit_ok'], d['ans_total']):.3f} | {rate(d['must_ok'], d['ans_total']):.3f} | "
            f"{rate(d['num_ok'], d['ans_total']):.3f} |"
        )
    report.append("")

    report.append("## Breakdown By Tags")
    report.append("| tag | total | not_found_accuracy | citations_present_rate | must_contain_pass_rate | numbers_grounded_rate |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for tag in sorted(by_tag.keys()):
        t = by_tag[tag]
        report.append(
            f"| {tag} | {t['total']} | {rate(t['nf_correct'], t['nf_total']):.3f} | "
            f"{rate(t['cit_ok'], t['ans_total']):.3f} | {rate(t['must_ok'], t['ans_total']):.3f} | "
            f"{rate(t['num_ok'], t['ans_total']):.3f} |"
        )
    report.append("")

    report.append("## Top-10 Failures")
    report.append("| case_id | reason | query |")
    report.append("|---|---|---|")
    for f in failures[:10]:
        report.append(f"| {f['case_id']} | {f['reason']} | {f['query']} |")

    out_path = Path("rag/bench/out/bench_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
