"""
PDF → Markdown Converter
Downloads a PDF from a URL and converts it to a clean .md file for RAG/LLM ingestion.

Requirements:
    uv pip install httpx pymupdf
"""

import httpx
import fitz  # pymupdf
from pathlib import Path

# ==========================================
# 1. CONFIGURATION — Edit this section only
# ==========================================

# Direct URL to the PDF you want to convert.
# Can be any publicly accessible PDF link.
PDF_URL = "https://www.depts.ttu.edu/research/tx-water-consortium/TXPWCFINALDRAFT.pdf"

# Where to save the output .md file.
OUTPUT_DIR = Path("./crawled-docs")

# What to name the output file (without extension).
# Leave as None to auto-generate from the PDF filename.
OUTPUT_FILENAME = None

# ==========================================
# 2. SCRIPT LOGIC — No edits needed below
# ==========================================

def pdf_bytes_to_markdown(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes and formats it as Markdown."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"<!-- Page {page_num} -->\n\n{text}")

    doc.close()
    return "\n\n---\n\n".join(pages)


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Auto-generate filename from URL if not specified
    if OUTPUT_FILENAME:
        filename = f"{OUTPUT_FILENAME}.md"
    else:
        pdf_name = PDF_URL.split("/")[-1].split("?")[0].replace(".pdf", "")
        filename = f"{pdf_name}.md"

    print(f"📥 Downloading: {PDF_URL}")

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        response = client.get(PDF_URL)
        response.raise_for_status()
        pdf_bytes = response.content

    print(f"✅ Downloaded ({len(pdf_bytes):,} bytes) — extracting text...")

    markdown = pdf_bytes_to_markdown(pdf_bytes)

    if not markdown.strip():
        print("⚠️  No text extracted. The PDF may be scanned/image-based.")
        print("   For scanned PDFs you'll need OCR — install: uv pip install pytesseract pillow")
        return

    output_path = OUTPUT_DIR / filename
    output_path.write_text(markdown, encoding="utf-8")

    print(f"✅ Saved: {output_path.absolute()}")
    print(f"   {len(markdown):,} characters across {markdown.count('<!-- Page')} pages")


if __name__ == "__main__":
    run()
