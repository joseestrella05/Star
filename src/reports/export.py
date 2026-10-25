"""Report export: edited Markdown -> standalone HTML -> PDF, with no network access."""
import html

import markdown

CSS = """
@page { size: A4; margin: 22mm 18mm 20mm; @bottom-center { content: element(footer); } }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1f2b; }
h1 { font-size: 17pt; margin: 0 0 14pt; padding-bottom: 6pt; border-bottom: 2px solid #1a1f2b; }
table { border-collapse: collapse; width: 100%; margin: 0 0 14pt; }
th, td { padding: 4pt 8pt; border-bottom: 1px solid #d5d9e0; text-align: left; }
th { background: #f1f3f6; }
td:last-child, th:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.footer { position: running(footer); font-size: 8pt; color: #6b7280; }
"""


def to_html(markdown_text, footer=""):
    # Escape first: user-edited Markdown must not inject raw HTML (scripts, remote images).
    body = markdown.markdown(html.escape(markdown_text, quote=False), extensions=["tables"])
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>Star report</title>'
            f'<style>{CSS}</style></head><body><div class="footer">{html.escape(footer)}</div>{body}</body></html>')


def no_fetch(url, *args, **kwargs):
    """WeasyPrint URL fetcher that refuses everything: rendering never touches the network or disk."""
    raise ValueError(f"external resources are disabled: {url}")


def to_pdf(markdown_text, footer=""):
    from weasyprint import HTML  # lazy: needs the system Pango library, absent on some hosts

    return HTML(string=to_html(markdown_text, footer), url_fetcher=no_fetch).write_pdf()
