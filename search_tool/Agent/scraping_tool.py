"""
Web scraping tool with Chromium as the primary method and aiohttp as a fast fallback.
"""

import logging
import asyncio
import platform
from typing import List, Type, Optional
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime

# Apply the asyncio policy fix at the top level
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def get_default_tags() -> List[str]:
    """Get default HTML tags for web scraping"""
    return ["p", "li", "div", "a", "span", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "code", "table"]
class WebScrapingInput(BaseModel):
    url: str = Field(description="URL to scrape")
    tags_to_extract: List[str] = Field(
        default_factory=get_default_tags, description="HTML tags to extract"
    )
    method: Optional[str] = Field(
        default="auto", 
        description="Scraping method: 'auto' (default), 'chromium', or 'aiohttp'"
    )
class WebScrapingTool(BaseTool):
    """
    Scrapes a website for its content. It first tries using a full browser (Chromium) 
    to handle JavaScript-heavy pages, and falls back to a faster, simpler method (aiohttp) 
    if the browser fails or for simple pages.
    """
    name: str = "scrape_website"
    description: str = """Scrape website content. Use this after a search tool has provided a promising URL.
    This tool is powerful and can read content from complex websites.
    PARAMETERS:
    - url (string): The complete URL to scrape.
    - method (string, optional): 'auto', 'chromium', or 'aiohttp'. 'auto' is recommended.
    """
    args_schema: Type[BaseModel] = WebScrapingInput
    max_content_length: int = Field(default=20000, exclude=True)
    timeout: int = Field(default=20, exclude=True)
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        exclude=True
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    def __init__(self, max_content_length: int = 20000, timeout: int = 20):
        super().__init__(
            max_content_length=max_content_length,
            timeout=timeout,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            args_schema=WebScrapingInput
        )
    async def _scrape_with_chromium(self, url: str, tags_to_extract: List[str]) -> str:
        """Scrape using a full browser (Chromium). Best for JS-heavy sites."""
        logger.info(f"🌐 Attempting Chromium scrape for: {url}")
        loader = AsyncChromiumLoader([url])
        html_docs = await asyncio.to_thread(loader.load)
        if not html_docs:
            raise Exception("Chromium loader failed to get any documents.")
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = await asyncio.to_thread(
            bs_transformer.transform_documents,
            html_docs,
            tags_to_extract=tags_to_extract,
        )
        if not docs_transformed:
            raise Exception("BeautifulSoupTransformer failed to extract content.")
        return docs_transformed[0].page_content
    async def _scrape_with_aiohttp(self, url: str, tags_to_extract: List[str]) -> str:
        """Scrape using a fast, simple HTTP request. Good for simple HTML pages."""
        logger.info(f"🔄 Attempting aiohttp scrape for: {url}")
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {'User-Agent': self.user_agent}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as response:
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                html_content = await response.text()
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        return soup.get_text(separator='\n', strip=True)
    async def _process_scraping(
        self, url: str, tags_to_extract: List[str] = None, method: str = "auto"
    ) -> str:
        """Orchestrates the scraping process with fallbacks."""
        if tags_to_extract is None:
            tags_to_extract = get_default_tags()

        content = ""
        used_method = ""
        start_time = datetime.now()

        try:
            # The "auto" method logic
            if method == 'auto':
                try:
                    content = await self._scrape_with_chromium(url, tags_to_extract)
                    used_method = "chromium"
                    logger.info("✅ Chromium scraping successful.")
                except Exception as e:
                    logger.warning(f"⚠️ Chromium scraping failed ({e}), falling back to aiohttp.")
                    content = await self._scrape_with_aiohttp(url, tags_to_extract)
                    used_method = "aiohttp"
                    logger.info("✅ aiohttp fallback successful.")
            
            # Logic for forcing a specific method
            elif method == 'chromium':
                content = await self._scrape_with_chromium(url, tags_to_extract)
                used_method = "chromium"
            elif method == 'aiohttp':
                content = await self._scrape_with_aiohttp(url, tags_to_extract)
                used_method = "aiohttp"
            else:
                raise ValueError(f"Unknown method: {method}")
            if not content:
                return f"❌ No content extracted from {url} using method {used_method}."
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "\n\n... (content truncated)"
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            return f"""
**Website Scraped:** {url}
**Method Used:** {used_method}
**Duration:** {duration:.2f} seconds
**Content Length:** {len(content)} characters
**Content Extracted:**
{content}
"""
        except Exception as e:
            return f"❌ Web scraping error for {url}: {str(e)}"
    def _run(self, url: str, tags_to_extract: List[str] = None, method: str = "auto") -> str:
        """Synchronous wrapper for LangChain."""
        return asyncio.run(self._process_scraping(url, tags_to_extract, method))
    async def _arun(self, url: str, tags_to_extract: List[str] = None, method: str = "auto") -> str:
        """Asynchronous entry point for LangChain."""
        return await self._process_scraping(url, tags_to_extract, method)