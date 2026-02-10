import os
from utils import load_jsonl

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>RAG Chunks Viewer</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.chunk {{ border: 1px solid #ccc; padding: 12px; margin-bottom: 16px; }}
.meta {{ color: #555; font-size: 12px; margin-bottom: 8px; }}
.text {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>RAG chunks</h1>
{content}
</body>
</html>
"""

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    chunks_path = os.path.join(project_dir, "data", "chunks", "chunks.jsonl")
    out_path = os.path.join(project_dir, "data", "chunks", "chunks.html")

    chunks = load_jsonl(chunks_path)

    blocks = []
    for c in chunks:
        meta = (
            f"chunk_id={c['chunk_id']} | "
            f"doc_id={c['doc_id']} | "
            f"chars={c['char_len']} | "
            f"pages={c['page_min']}–{c['page_max']} | "
            f"paras={c['para_min']}–{c['para_max']}"
        )
        blocks.append(
            f"<div class='chunk'>"
            f"<div class='meta'>{meta}</div>"
            f"<div class='text'>{c['text']}</div>"
            f"</div>"
        )

    html = HTML_TEMPLATE.format(content="\n".join(blocks))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Written: {out_path}")

if __name__ == "__main__":
    main()
