"""Web tools: search and fetch."""

import logging

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web using DuckDuckGo and return results.",
            parameters=[
                ToolParameter(name="query", type=ToolParameterType.STRING,
                              description="Search query"),
                ToolParameter(name="max_results", type=ToolParameterType.INTEGER,
                              description="Maximum number of results (default 5)", required=False),
            ],
        )

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> str:
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}\n")

            if not results:
                return f"No results found for: {query}"
            return "\n---\n".join(results)
        except ImportError:
            return "Error: duckduckgo-search package not installed."
        except Exception as e:
            return f"Search error: {e}"


class WebFetchTool(Tool):
    def __init__(self, max_size: int = 524288, timeout: int = 15):
        self.max_size = max_size
        self.timeout = timeout

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_fetch",
            description="Fetch and extract text content from a web URL.",
            parameters=[
                ToolParameter(name="url", type=ToolParameterType.STRING,
                              description="URL to fetch"),
            ],
        )

    async def execute(self, url: str, **kwargs) -> str:
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.timeout,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove scripts and styles
                    for element in soup(["script", "style", "nav", "footer", "header"]):
                        element.decompose()

                    text = soup.get_text(separator="\n", strip=True)
                else:
                    text = response.text

                # Truncate if too large
                if len(text) > self.max_size:
                    text = text[:self.max_size] + "\n\n[Content truncated]"

                return text

        except ImportError:
            return "Error: httpx and/or beautifulsoup4 packages not installed."
        except Exception as e:
            return f"Fetch error: {e}"
