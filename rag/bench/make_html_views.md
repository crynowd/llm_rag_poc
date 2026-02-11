format_version: 1

JSONL to HTML viewer

This utility converts a JSONL file into a single self-contained HTML file
for human review. It supports search, filters, and expandable details.

Examples (do not run here):

```bash
python rag/bench/jsonl_to_html.py --in rag/bench/bench_cases_normalized.jsonl
python rag/bench/jsonl_to_html.py --in rag/bench/out/bench_run_result.jsonl
```
