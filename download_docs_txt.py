"""
Universal Documentation Crawler using Crawl4AI
Optimized for: Technical Docs, RAG/LLM Ingestion, and Headless Environments.

Requirements:
    uv pip install crawl4ai playwright httpx
    python -m playwright install chromium
"""

import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    BFSDeepCrawlStrategy,
    DefaultMarkdownGenerator,
    FilterChain,
    URLPatternFilter,
    PruningContentFilter,
)

# ==========================================
# 1. CONFIGURATION — Edit this section only
# ==========================================

# List of URLs to start crawling from.
BASE_URLS = [
    "https://www.depts.ttu.edu/research/tx-water-consortium/TXPWCFINALDRAFT.pdf",
]

# Where to save the resulting .md files on your machine.
OUTPUT_DIR = Path("./crawled-docs")

# How many "clicks deep" from the BASE_URL to follow.
# 1-2 = shallow, 3-4 = medium, 5+ = deep
MAX_DEPTH = 3

# Glob pattern to restrict which URLs get crawled.
# Use ** to match any number of path segments.
ALLOWED_URL_PATTERN = "www.depts.ttu.edu/**"

# CSS selector for the main content area.
# Set to None to grab the full page (safest default if you're getting empty results).
CSS_SELECTOR = None

# Content pruning threshold.
# 0.50 = strict, 0.42 = standard, 0.35 = lenient (best for legal/regulatory/tables)
# Set to None to disable pruning entirely — recommended if output is empty.
CONTENT_THRESHOLD = None

# ==========================================
# 2. CRAWLER LOGIC — No edits needed below
# ==========================================

async def crawl_site(crawler: AsyncWebCrawler, base_url: str, seen_urls: set, saved_count_ref: list):
    """Crawls a single base URL and saves results as Markdown files."""

    url_filter = FilterChain(filters=[
        URLPatternFilter(
            patterns=[ALLOWED_URL_PATTERN],
            use_glob=True,
            reverse=False
        ),
    ])

    deep_crawl = BFSDeepCrawlStrategy(
        max_depth=MAX_DEPTH,
        filter_chain=url_filter,
        include_external=False,
    )

    # Build markdown generator — with or without content pruning
    if CONTENT_THRESHOLD is not None:
        md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=CONTENT_THRESHOLD,
                threshold_type="fixed",
            )
        )
    else:
        md_generator = DefaultMarkdownGenerator()

    crawl_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        css_selector=CSS_SELECTOR,
        deep_crawl_strategy=deep_crawl,
        page_timeout=30000,      # 30s timeout for slow/JS-heavy pages
        verbose=True,
    )

    print(f"\n🌐 Crawling: {base_url}")
    results = await crawler.arun(base_url, config=crawl_config)

    if not isinstance(results, list):
        results = [results]

    pdf_links = []

    for i, result in enumerate(results):
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)

        if not result.success:
            print(f"  ⚠️  Failed: {result.url}")
            continue

        if not result.markdown:
            print(f"  ⚠️  No markdown: {result.url}")
            continue

        md_obj = result.markdown

        # Prefer fit_markdown (pruned) if available and non-empty, else fall back to raw
        content = (md_obj.fit_markdown or "").strip() or (md_obj.raw_markdown or "").strip()

        if not content:
            print(f"  ⚠️  Empty content: {result.url}")
            continue

        # Collect PDF links for later download
        pdf_links += re.findall(r'https?://[^\s"\'<>]+\.pdf', content)

        # Build filename from URL path
        parsed = urlparse(result.url)
        slug = parsed.path.strip("/").replace("/", "_")
        filename = f"{slug}.md" if slug else "index.md"

        filepath = OUTPUT_DIR / filename
        filepath.write_text(content, encoding="utf-8")
        saved_count_ref[0] += 1
        print(f"  [{saved_count_ref[0]:3d}] ✅ Saved: {filename}  ({len(content):,} chars)")

    return pdf_links


async def download_pdfs(pdf_links: list):
    """Downloads any PDFs found during the crawl."""
    pdf_dir = OUTPUT_DIR / "pdfs"
    unique_pdfs = list(set(pdf_links))

    if not unique_pdfs:
        return

    pdf_dir.mkdir(exist_ok=True)
    print(f"\n📄 Downloading {len(unique_pdfs)} PDF(s)...")

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for pdf_url in unique_pdfs:
            try:
                response = await client.get(pdf_url)
                pdf_name = pdf_url.split("/")[-1].split("?")[0]
                (pdf_dir / pdf_name).write_bytes(response.content)
                print(f"  ✅ {pdf_name}")
            except Exception as e:
                print(f"  ❌ Failed: {pdf_url} — {e}")


async def run_crawl():
    """Main entry point."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        java_script_enabled=True,
    )

    seen_urls = set()
    saved_count_ref = [0]
    all_pdf_links = []

    print(f"🚀 Starting crawl across {len(BASE_URLS)} site(s)")
    print(f"📂 Output: {OUTPUT_DIR.absolute()}\n")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for base_url in BASE_URLS:
            pdf_links = await crawl_site(crawler, base_url, seen_urls, saved_count_ref)
            all_pdf_links.extend(pdf_links)

    await download_pdfs(all_pdf_links)

    print(f"\n{'='*50}")
    print(f"✅ Done! Saved {saved_count_ref[0]} pages to '{OUTPUT_DIR}'")
    if all_pdf_links:
        print(f"📁 PDFs saved to '{OUTPUT_DIR / 'pdfs'}'")

    if saved_count_ref[0] == 0:
        print("\n⚠️  Nothing was saved. Try these fixes in order:")
        print("   1. Check that BASE_URLS is correct and the site is publicly accessible")
        print("   2. Broaden ALLOWED_URL_PATTERN (e.g. 'https://yourdomain.com/**')")
        print("   3. CONTENT_THRESHOLD is already None — good")
        print("   4. CSS_SELECTOR is already None — good")
        print("   5. The site may be blocking bots — try adding a user_agent to BrowserConfig")


if __name__ == "__main__":
    asyncio.run(run_crawl())
