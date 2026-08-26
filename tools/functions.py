
import html2text
import httpx
from loguru import logger
from playwright.async_api import async_playwright
from core.database import AsyncSessionLocal
from core.memory import semantic_search


async def get_webpage_content(url: str) -> str:
    logger.info(f"Tool executed: Fetching content from {url}")

    try:

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status() 

            return response.text

    except Exception as e:
        logger.error(f"Failed to fetch {url}: {str(e)}")
        return f"Error fetching webpage: {str(e)}"

async def calculate_string_length(text: str) -> str:
    logger.info("Tool executed: Calculating string length")
    return str(len(text))

async def extract_markdown_from_url(url: str) -> str:
    logger.info(f"Tool executed: Extracting markdown content from {url}")

    try:
        async with async_playwright() as p:

            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            html_content = await page.content()
            await browser.close()

            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True

            markdown_content = h.handle(html_content)
            return markdown_content[:8000]

    except Exception as e:
        logger.error(f"Playwright extraction failed: {str(e)}")
        return f"Error extracting content: {str(e)}"

async def search_memory(query: str) -> str:
    """Queries long-term semantic memory for relevant past observations"""
    async with AsyncSessionLocal() as session:
        results = await semantic_search(query = query, db=session, top_k=3)
        if not results:
            return "No relevant memories found."

        formatted = [f"- {entry.content}" for entry in results]
        return "Relevant memories found:\n" + "\n".join(formatted)