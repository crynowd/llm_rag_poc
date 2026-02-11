#!/usr/bin/env python3
"""
JSONL -> HTML viewer (stdlib only).
format_version: 1
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: Path, limit: Optional[int]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total_lines += 1
            if limit is not None and len(rows) >= limit:
                continue
            try:
                obj = json.loads(line)
                rows.append({"_parse_error": False, "_raw": line, "data": obj})
            except json.JSONDecodeError:
                rows.append({"_parse_error": True, "_raw": line, "data": None})
    return {"rows": rows, "total_lines": total_lines}


def build_html(
    rows: List[Dict[str, Any]],
    total_lines: int,
    title: str,
    source_name: str,
) -> str:
    payload = {
        "meta": {
            "title": title,
            "source": source_name,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "total_lines": total_lines,
            "rows_shown": len(rows),
        },
        "rows": rows,
    }

    data_json = json.dumps(payload, ensure_ascii=False)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__TITLE__</title>
  <style>
    :root {{
      --bg: #f8f9fb;
      --fg: #111;
      --muted: #6b7280;
      --ok: #0f7b0f;
      --fail: #b00020;
      --warn: #f59e0b;
      --nf: #1d4ed8;
      --border: #e5e7eb;
      --panel: #ffffff;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: var(--bg);
      color: var(--fg);
    }}
    header {{
      padding: 16px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{
      margin: 0 0 6px 0;
      font-size: 20px;
    }}
    header .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .filters {{
      padding: 12px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .filters input, .filters select {{
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      font-size: 13px;
    }}
    .content {{
      padding: 12px 20px 24px 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{
      text-align: left;
      background: #f3f4f6;
      font-weight: 600;
    }}
    tr.warn {{
      background: #fff7ed;
    }}
    .status-ok {{
      color: var(--ok);
      font-weight: 700;
    }}
    .status-fail {{
      color: var(--fail);
      font-weight: 700;
    }}
    .type-nf {{
      color: var(--nf);
      font-weight: 700;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    details {{
      margin-top: 6px;
    }}
    summary {{
      cursor: pointer;
      color: #2563eb;
    }}
    .subtable {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 6px;
    }}
    .subtable th, .subtable td {{
      border: 1px solid var(--border);
      padding: 4px 6px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .nowrap {{
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <header>
    <h1 id="title"></h1>
    <div class="meta" id="meta"></div>
  </header>
  <div class="filters">
    <input id="q" type="text" placeholder="Search substring (all fields)"/>
    <select id="docFilter"><option value="">doc_id: all</option></select>
    <select id="expFilter"><option value="">expected.type: all</option></select>
    <select id="statusFilter"><option value="">status: all</option></select>
    <span class="muted" id="count"></span>
  </div>
  <div class="content">
    <table>
      <thead id="thead"></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <script>
    const DATA = __DATA_JSON__;
    const rows = DATA.rows;
    const meta = DATA.meta;

    const title = document.getElementById("title");
    const metaEl = document.getElementById("meta");
    title.textContent = meta.title;
    metaEl.textContent = `source=${meta.source} | lines=${meta.total_lines} | shown=${meta.rows_shown} | generated=${meta.generated_utc}`;

    function getField(row, path) {{
      const parts = path.split(".");
      let cur = row;
      for (const p of parts) {{
        if (!cur || typeof cur !== "object" || !(p in cur)) return undefined;
        cur = cur[p];
      }}
      return cur;
    }}

    function pickData(r) {{
      return r.data || {{}};
    }}

    const hasAny = {{
      case_id: rows.some(r => getField(pickData(r), "case_id")),
      query: rows.some(r => getField(pickData(r), "query")),
      doc_id: rows.some(r => getField(pickData(r), "doc_id")),
      expected: rows.some(r => getField(pickData(r), "expected")),
      run: rows.some(r => getField(pickData(r), "run")),
      status: rows.some(r => getField(pickData(r), "status")),
      timing_ms: rows.some(r => getField(pickData(r), "timing_ms") !== undefined),
    }};

    const useDetailed = hasAny.case_id || hasAny.query || hasAny.doc_id || hasAny.expected || hasAny.run || hasAny.status || hasAny.timing_ms;

    function buildFilters() {{
      const docSet = new Set();
      const expSet = new Set();
      const statusSet = new Set();
      rows.forEach(r => {{
        const d = pickData(r);
        if (d.doc_id) docSet.add(d.doc_id);
        if (d.expected && d.expected.type) expSet.add(d.expected.type);
        if (d.status) statusSet.add(d.status);
      }});
      const docFilter = document.getElementById("docFilter");
      const expFilter = document.getElementById("expFilter");
      const statusFilter = document.getElementById("statusFilter");
      [...docSet].sort().forEach(v => docFilter.appendChild(new Option(v, v)));
      [...expSet].sort().forEach(v => expFilter.appendChild(new Option(v, v)));
      [...statusSet].sort().forEach(v => statusFilter.appendChild(new Option(v, v)));
    }}

    function makeSubtable(headers, rows) {{
      if (!rows || !rows.length) return "";
      let html = "<table class='subtable'><thead><tr>";
      headers.forEach(h => html += `<th>${{h}}</th>`);
      html += "</tr></thead><tbody>";
      rows.forEach(r => {{
        html += "<tr>";
        headers.forEach(h => html += `<td>${{r[h] ?? ""}}</td>`);
        html += "</tr>";
      }});
      html += "</tbody></table>";
      return html;
    }}

    function renderRowDetailed(r, idx) {{
      const d = pickData(r);
      const exp = d.expected || {{}};
      const run = d.run || {{}};
      const answer = run.answer_text || "";
      const isNF = run.is_not_found;
      const expType = exp.type;
      const hasNF = answer.includes("NOT_FOUND");
      const mismatch = (hasNF && isNF === false) || (!hasNF && isNF === true);
      const status = d.status || (run.stop_reason === "error" ? "FAIL" : "");

      const tr = document.createElement("tr");
      if (mismatch) tr.classList.add("warn");

      const cells = [];
      cells.push(`<td class="nowrap">${{idx}}</td>`);
      if (hasAny.case_id) cells.push(`<td class="mono">${{d.case_id ?? ""}}</td>`);
      if (hasAny.query) cells.push(`<td>${{d.query ?? ""}}</td>`);
      if (hasAny.doc_id) cells.push(`<td class="mono">${{d.doc_id ?? ""}}</td>`);
      if (hasAny.expected) {{
        const typeClass = expType === "not_found" ? "type-nf" : "";
        const must = (exp.must_contain || []).map(x => `<li>${{x}}</li>`).join("");
        const reqInf = exp.requires_inference === true ? "true" : "false";
        cells.push(
          `<td><div class="${{typeClass}}">type=${{expType ?? ""}}</div>` +
          `<div class="muted">requires_inference=${{reqInf}}</div>` +
          (must ? `<details><summary>must_contain (${{(exp.must_contain || []).length}})</summary><ul>${{must}}</ul></details>` : "") +
          `</td>`
        );
      }}
      if (hasAny.run) {{
        const qrows = (run.quotes || []).map(q => ({
          "q_id": q.q_id || "",
          "doc_id": q.source ? q.source.doc_id : "",
          "chunk_id": q.chunk_id || "",
          "quote": q.quote || "",
        }));
        const rrows = (run.retrieved || []).map(x => ({
          "rank": x.rank ?? "",
          "similarity": x.similarity ?? "",
          "doc_id": x.doc_id ?? "",
          "chunk_id": x.chunk_id ?? "",
        }));
        const warn = mismatch ? "<div class='muted' style='color:var(--warn)'>warning: NOT_FOUND mismatch</div>" : "";
        const ansBlock = `<div class="mono">${{answer}}</div>`;
        const qTable = makeSubtable(["q_id","doc_id","chunk_id","quote"], qrows);
        const rTable = makeSubtable(["rank","similarity","doc_id","chunk_id"], rrows);
        cells.push(
          `<td>${{ansBlock}}` +
          `${{warn}}` +
          (qrows.length ? `<details><summary>quotes (${{qrows.length}})</summary>${{qTable}}</details>` : "") +
          (rrows.length ? `<details><summary>retrieved (${{rrows.length}})</summary>${{rTable}}</details>` : "") +
          `</td>`
        );
      }}
      if (hasAny.status) {{
        const cls = status === "FAIL" ? "status-fail" : (status === "OK" ? "status-ok" : "");
        cells.push(`<td class="${{cls}}">${{status}}</td>`);
      }}
      if (hasAny.timing_ms) {{
        cells.push(`<td class="mono">${{d.timing_ms ?? ""}}</td>`);
      }}

      const raw = r._parse_error ? r._raw : JSON.stringify(d, null, 2);
      const rawLabel = r._parse_error ? "PARSE_ERROR" : "Raw JSON";
      cells.push(`<td><details><summary>${{rawLabel}}</summary><pre class="mono">${{raw}}</pre></details></td>`);

      tr.innerHTML = cells.join("");
      return tr;
    }}

    function renderRowFallback(r, idx) {{
      const d = pickData(r);
      const keys = d ? Object.keys(d).join(", ") : "";
      const summary = d ? JSON.stringify(d).slice(0, 200) : "";
      const raw = r._parse_error ? r._raw : JSON.stringify(d, null, 2);
      const rawLabel = r._parse_error ? "PARSE_ERROR" : "Raw JSON";
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td class="nowrap">${{idx}}</td>` +
        `<td class="mono">${{keys}}</td>` +
        `<td class="mono">${{summary}}</td>` +
        `<td><details><summary>${{rawLabel}}</summary><pre class="mono">${{raw}}</pre></details></td>`;
      return tr;
    }}

    function buildTableHead() {{
      const thead = document.getElementById("thead");
      let cols = ["idx"];
      if (useDetailed) {{
        if (hasAny.case_id) cols.push("case_id");
        if (hasAny.query) cols.push("query");
        if (hasAny.doc_id) cols.push("doc_id");
        if (hasAny.expected) cols.push("expected");
        if (hasAny.run) cols.push("run");
        if (hasAny.status) cols.push("status");
        if (hasAny.timing_ms) cols.push("timing_ms");
        cols.push("json");
      }} else {{
        cols = ["idx", "keys", "short_summary", "json"];
      }}
      thead.innerHTML = "<tr>" + cols.map(c => `<th>${{c}}</th>`).join("") + "</tr>";
    }}

    function rowMatchesFilters(r, q, docId, expType, status) {{
      const d = pickData(r);
      if (docId && d.doc_id !== docId) return false;
      if (expType && (!d.expected || d.expected.type !== expType)) return false;
      if (status && d.status !== status) return false;
      if (q) {{
        const hay = JSON.stringify(d).toLowerCase();
        if (!hay.includes(q)) return false;
      }}
      return true;
    }}

    function render() {{
      const tbody = document.getElementById("tbody");
      const q = document.getElementById("q").value.trim().toLowerCase();
      const docId = document.getElementById("docFilter").value;
      const expType = document.getElementById("expFilter").value;
      const status = document.getElementById("statusFilter").value;

      tbody.innerHTML = "";
      let shown = 0;
      rows.forEach((r, idx) => {{
        if (!rowMatchesFilters(r, q, docId, expType, status)) return;
        shown += 1;
        const tr = useDetailed ? renderRowDetailed(r, shown) : renderRowFallback(r, shown);
        tbody.appendChild(tr);
      }});
      document.getElementById("count").textContent = `shown: ${shown}`;
    }}

    buildFilters();
    buildTableHead();
    render();

    ["q","docFilter","expFilter","statusFilter"].forEach(id => {{
      document.getElementById(id).addEventListener("input", render);
      document.getElementById(id).addEventListener("change", render);
    }});
  </script>
</body>
</html>"""

    return template.replace("__TITLE__", title).replace("__DATA_JSON__", data_json)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert JSONL to a readable HTML report")
    ap.add_argument("--in", dest="input_path", required=True, help="Path to JSONL")
    ap.add_argument("--out", dest="output_path", default=None, help="Path to HTML")
    ap.add_argument("--title", dest="title", default=None, help="Optional report title")
    ap.add_argument("--limit", dest="limit", type=int, default=None, help="Optional max rows")
    args = ap.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.output_path) if args.output_path else None
    if out_path is None:
        out_dir = Path("rag/bench/out")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (in_path.stem + ".html")

    data = read_jsonl(in_path, args.limit)
    title = args.title or f"JSONL View: {in_path.name}"
    html = build_html(
        rows=data["rows"],
        total_lines=data["total_lines"],
        title=title,
        source_name=in_path.name,
    )
    out_path.write_text(html, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
