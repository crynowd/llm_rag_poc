#!/usr/bin/env python3
"""Validate normalized benchmark cases JSONL.
format_version: 1
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ALLOWED_EXPECTED_TYPES = {"answer", "not_found"}
ALLOWED_DIFFICULTY = {1, 2, 3}


def validate_row(obj: dict, idx: int) -> list[str]:
    errors = []
    required_top = [
        "format_version",
        "case_id",
        "source_case_id",
        "doc_id",
        "query",
        "expected",
        "tags",
        "difficulty",
    ]
    for key in required_top:
        if key not in obj:
            errors.append(f"line {idx}: missing field '{key}'")

    if errors:
        return errors

    if obj["format_version"] != 1:
        errors.append(f"line {idx}: format_version must be 1")

    if not isinstance(obj["case_id"], str) or not obj["case_id"].strip():
        errors.append(f"line {idx}: case_id must be non-empty string")

    if obj["source_case_id"] is not None and not isinstance(obj["source_case_id"], str):
        errors.append(f"line {idx}: source_case_id must be string or null")

    if not isinstance(obj["doc_id"], str) or not obj["doc_id"].strip():
        errors.append(f"line {idx}: doc_id must be non-empty string")

    if not isinstance(obj["query"], str) or not obj["query"].strip():
        errors.append(f"line {idx}: query must be non-empty string")

    if not isinstance(obj["tags"], list) or not all(isinstance(x, str) for x in obj["tags"]):
        errors.append(f"line {idx}: tags must be array of strings")

    if obj["difficulty"] not in ALLOWED_DIFFICULTY:
        errors.append(f"line {idx}: difficulty must be one of 1,2,3")

    exp = obj["expected"]
    if not isinstance(exp, dict):
        errors.append(f"line {idx}: expected must be object")
        return errors

    for key in ["type", "must_contain", "forbidden", "requires_inference"]:
        if key not in exp:
            errors.append(f"line {idx}: expected missing '{key}'")

    if errors:
        return errors

    if exp["type"] not in ALLOWED_EXPECTED_TYPES:
        errors.append(f"line {idx}: expected.type must be answer|not_found")

    if not isinstance(exp["must_contain"], list) or not all(isinstance(x, str) for x in exp["must_contain"]):
        errors.append(f"line {idx}: expected.must_contain must be array of strings")

    if not isinstance(exp["forbidden"], list) or not all(isinstance(x, str) for x in exp["forbidden"]):
        errors.append(f"line {idx}: expected.forbidden must be array of strings")

    if not isinstance(exp["requires_inference"], bool):
        errors.append(f"line {idx}: expected.requires_inference must be bool")

    return errors


def run(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 2

    rows = []
    errors = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {idx}: invalid JSON: {e}")
                continue

            rows.append(obj)
            cid = obj.get("case_id")
            if isinstance(cid, str):
                if cid in seen_ids:
                    errors.append(f"line {idx}: duplicate case_id: {cid}")
                seen_ids.add(cid)

            errors.extend(validate_row(obj, idx))

    case_map = {r.get("case_id"): r for r in rows if isinstance(r.get("case_id"), str)}
    for r in rows:
        src = r.get("source_case_id")
        if src is None:
            continue
        if src not in case_map:
            errors.append(f"case_id={r.get('case_id')}: source_case_id points to missing case: {src}")
            continue
        base = case_map[src]
        if base.get("source_case_id") is not None:
            errors.append(
                f"case_id={r.get('case_id')}: source_case_id must reference base case (source_case_id=null): {src}"
            )

    total = len(rows)
    base_count = sum(1 for r in rows if r.get("source_case_id") is None)
    para_count = total - base_count
    not_found_count = sum(
        1
        for r in rows
        if isinstance(r.get("expected"), dict) and r["expected"].get("type") == "not_found"
    )

    expected_counter = Counter(
        r.get("expected", {}).get("type")
        for r in rows
        if isinstance(r.get("expected"), dict)
    )
    difficulty_counter = Counter(r.get("difficulty") for r in rows)
    tag_counter = Counter()
    for r in rows:
        tags = r.get("tags", [])
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, str):
                    tag_counter[t] += 1

    print(f"total_cases: {total}")
    print(f"base_cases: {base_count}")
    print(f"paraphrase_cases: {para_count}")
    print(f"not_found_cases: {not_found_count}")
    print(
        "count_by_expected_type: "
        + json.dumps(
            {k: expected_counter.get(k, 0) for k in sorted(ALLOWED_EXPECTED_TYPES)},
            ensure_ascii=False,
        )
    )
    print(
        "count_by_difficulty: "
        + json.dumps(
            {str(k): difficulty_counter.get(k, 0) for k in sorted(ALLOWED_DIFFICULTY)},
            ensure_ascii=False,
        )
    )
    print("top_tags_frequency:")
    for tag, cnt in tag_counter.most_common(10):
        print(f"- {tag}: {cnt}")

    if errors:
        print("validation: FAILED")
        for err in errors[:100]:
            print(f"- {err}")
        if len(errors) > 100:
            print(f"- ... and {len(errors) - 100} more errors")
        return 1

    print("validation: OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized benchmark cases JSONL")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to normalized JSONL")
    args = parser.parse_args()
    raise SystemExit(run(Path(args.input_path)))


if __name__ == "__main__":
    main()
