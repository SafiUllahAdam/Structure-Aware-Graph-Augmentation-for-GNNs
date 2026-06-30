"""Render a Markdown file to PDF (weasyprint, no Chromium). Usage: python docs/build_pdf.py docs/virgo_guide.md docs/virgo_guide.pdf"""
import sys, re, markdown
from weasyprint import HTML

src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

# emoji -> glyphs the bundled DejaVu fonts can draw (PDF has no emoji font)
for a, b in {"✅": "✔", "🔵": "●", "⏳": "•", "⭐": "★", "⬅️": "←", "️": ""}.items():
    text = text.replace(a, b)

# python-markdown needs a blank line before a list; insert one when a list follows a paragraph
list_re = re.compile(r"^\s*([-*+]|\d+\.)\s+\S")
lines, fixed = text.split("\n"), []
for ln in lines:
    if list_re.match(ln) and fixed and fixed[-1].strip() and not list_re.match(fixed[-1]):
        fixed.append("")
    fixed.append(ln)
text = "\n".join(fixed)

body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1f2937; }
h1 { font-size: 22pt; } h2 { font-size: 16pt; margin-top: 1.2em; border-bottom: 1px solid #ddd; padding-bottom: 2px; }
h3 { font-size: 13pt; }
code { font-family: 'DejaVu Sans Mono', monospace; background: #f3f4f6; padding: 1px 3px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f3f4f6; padding: 8px; border-radius: 5px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; }
th, td { border: 1px solid #cbd5e1; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef2ff; }
blockquote { border-left: 4px solid #c7d2fe; margin: 0; padding: 0.2em 1em; background: #f8fafc; color: #374151; }
li { margin: 0.15em 0; }
a { color: #2563eb; }
"""

html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
HTML(string=html).write_pdf(out)
print("wrote", out)
