"""
Tavily search tool for domain-specific web search
"""

import logging
from typing import List, Optional, Type, Any
import asyncio
from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def create_summarizer_llm(google_api_key: str) -> ChatGoogleGenerativeAI:
    """Create summarization LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.1,
        max_tokens=1000,
        google_api_key=google_api_key,
    )
def format_search_results(results: List[dict], max_content_size: int) -> List[str]:
    """Format search results into readable strings"""
    formatted_results = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        content = result.get("content", "No content available")
        logger.info(f"📄 Processing result {i}: {title[:50]}...")
        if len(content) > max_content_size:
            content = content[:max_content_size] + "..."
        formatted_result = f"""
Result {i}:
Title: {title}
URL: {url}
Content: {content}
---
"""
        formatted_results.append(formatted_result)
    return formatted_results
def create_summary_prompt(search_results: str, original_query: str) -> str:
    """Create prompt for result summarization"""
    return f"""
You are a technical documentation summarizer. Your job is to extract and summarize only the most relevant information from search results.
Original User Query: "{original_query}"
Search Results to Summarize:
{search_results}
Instructions:
1. Focus ONLY on information directly relevant to answering the user's query
2. Remove redundant content, boilerplate text, and navigation elements  
3. Preserve specific technical details, code examples, and step-by-step instructions
4. Maintain source URLs for attribution
5. Keep the summary comprehensive but concise
6. Format clearly for easy reading
Relevant Summary:
"""
class TavilySearchInput(BaseModel):
    query: str = Field(description="Search query with relevant keywords")
    sites: List[str] = Field(
        description="Website domains to search (e.g., ['docs.langchain.com'])"
    )
    max_results: Optional[int] = Field(
        default=None, description="Maximum results to return"
    )
    depth: Optional[str] = Field(
        default=None, description="Search depth: 'basic' or 'advanced'"
    )
class TavilyDomainSearchTool(BaseTool):
    """Search specific domains using Tavily"""
    name: str = "search_documentation"
    description: str = """Search documentation websites using Tavily web search.
    REQUIRED PARAMETERS:
    - query (string): Search query with relevant keywords - what you want to find
    - sites (list): Website domains to search within (e.g., ['docs.langchain.com', 'fastapi.tiangolo.com'])
    OPTIONAL PARAMETERS:
    - max_results (integer): Maximum number of search results to return (default: 10)
    - depth (string): Search depth - 'basic' for quick searches or 'advanced' for comprehensive searches (default: 'basic')
    Usage Guidelines:
    1. Create keyword-rich search query from user's question
    2. Select relevant website domains based on technologies mentioned
    3. Use 'basic' depth for quick answers, 'advanced' for thorough research
    4. Adjust max_results based on how comprehensive you need the answer to be
    Examples:
    - Quick search: query="LangChain custom tools", sites=["docs.langchain.com"], depth="basic", max_results=5
    - Comprehensive search: query="FastAPI authentication middleware", sites=["fastapi.tiangolo.com"], depth="advanced", max_results=15
    Best Practices:
    - Include technical terms and framework names in queries
    - Choose appropriate domains for the question context
    - Prefer official documentation sites over third-party sources
    - Use specific queries rather than broad terms for better results
    """
    args_schema: Type[BaseModel] = TavilySearchInput
    tavily_client: Any = Field(default=None, exclude=True)
    api_key: str = Field(exclude=True)
    default_max_results: int = Field(default=10, exclude=True)
    default_depth: str = Field(default="basic", exclude=True)
    max_content_size: int = Field(default=10000, exclude=True)
    enable_summarization: bool = Field(default=False, exclude=True)
    summarizer_llm: Any = Field(default=None, exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    def __init__(
        self,
        api_key: str,
        max_results: int = 10,
        depth: str = "basic",
        max_content_size: int = 10000,
        enable_summarization: bool = False,
        google_api_key: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key,
            default_max_results=max_results,
            default_depth=depth,
            max_content_size=max_content_size,
            enable_summarization=enable_summarization,
            args_schema=TavilySearchInput,
        )

        if not api_key:
            raise ValueError("TAVILY_API_KEY is required")
        object.__setattr__(self, "tavily_client", TavilyClient(api_key=api_key))
        if enable_summarization and google_api_key:
            summarizer = create_summarizer_llm(google_api_key)
            object.__setattr__(self, "summarizer_llm", summarizer)
            logger.info("🧠 Search result summarization enabled with Gemini Flash-Lite")
        elif enable_summarization:
            logger.warning("⚠️ Summarization disabled: google_api_key not provided")
            object.__setattr__(self, "enable_summarization", False)
        logger.info(
            f"Tavily search tool initialized (summarization: {'enabled' if self.enable_summarization else 'disabled'})"
        )
    async def _search_async(self, query: str, sites: List[str], max_results: int = None, depth: str = None) -> str:
        """Execute search with given parameters asynchronously"""
        try:
            final_max_results = max_results or self.default_max_results
            final_depth = depth or self.default_depth

            logger.info(f"🔍 Searching: '{query}' on sites: {sites}")
            logger.info(
                f"📊 Parameters: max_results={final_max_results}, depth={final_depth}"
            )
            search_results = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                max_results=final_max_results,
                search_depth=final_depth,
                include_domains=sites,
            )
            logger.info(f"📥 Received {len(search_results.get('results', []))} results")
            if not search_results.get("results"):
                logger.warning("⚠️ No search results returned")
                return "No results found. Try a different search query or check if domains are accessible."
            formatted_results = format_search_results(
                search_results["results"][:final_max_results], self.max_content_size
            )
            final_result = "\n".join(formatted_results)

            logger.info(
                f"✅ Processed {len(search_results['results'])} results, returning {len(final_result)} characters"
            )

            if self.enable_summarization and self.summarizer_llm:
                try:
                    logger.info("🧠 Summarizing results...")
                    summarized_result = await self._summarize_results_async(final_result, query)
                    reduction = round(
                        (1 - len(summarized_result) / len(final_result)) * 100
                    )
                    logger.info(
                        f"📊 Summarization: {len(final_result)} → {len(summarized_result)} chars ({reduction}% reduction)"
                    )
                    return summarized_result
                except Exception as e:
                    logger.error(
                        f"❌ Summarization failed: {e}. Returning original results."
                    )
            return final_result
        except Exception as e:
            error_msg = f"❌ Search error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    async def _summarize_results_async(self, search_results: str, original_query: str) -> str:
        """Summarize search results using LLM asynchronously"""
        try:
            prompt = create_summary_prompt(search_results, original_query)
            response = await asyncio.to_thread(self.summarizer_llm.invoke, prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return search_results
    def _run(
        self, query: str, sites: List[str], max_results: int = None, depth: str = None
    ) -> str:
        """Execute search with given parameters"""
        return asyncio.run(self._search_async(query, sites, max_results, depth))
    async def _arun(
        self, query: str, sites: List[str], max_results: int = None, depth: str = None
    ) -> str:
        """Async version of search"""
        logger.info(f"🔍 Async search: '{query}'")
        return await self._search_async(query, sites, max_results, depth)
