"""Web tools: search and fetch."""

import ipaddress
import logging
from urllib.parse import urlparse

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


def _is_ssrf_target(url: str) -> str | None:
    """Block URLs targeting private/internal resources. Returns error or None."""
    parsed = urlparse(url)

    # Block non-HTTP schemes
    if parsed.scheme not in ("http", "https"):
        return f"Error: Only http/https URLs are allowed (got '{parsed.scheme}')."

    hostname = parsed.hostname or ""

    # Block cloud metadata endpoints
    if hostname in ("169.254.169.254", "metadata.google.internal"):
        return "Error: Access to cloud metadata endpoints is blocked."

    # Block localhost variants
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return "Error: Access to localhost is blocked."

    # Block private IP ranges
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return f"Error: Access to private/internal IP {hostname} is blocked."
    except ValueError:
        pass  # hostname is a domain name, not an IP — that's fine

    return None


class WebSearchTool(Tool):
    read_only = True

    def __init__(self, tavily_api_key: str | None = None):
        self._tavily_api_key = tavily_api_key

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description=(
                "Search the web and return results.\n\n"
                "When to use:\n"
                "- User asks a factual question you're not confident about\n"
                "- User explicitly asks you to search or look something up\n"
                "- You need current information (news, docs, releases, prices)\n\n"
                "When NOT to use:\n"
                "- Questions you can answer from memory or general knowledge\n"
                "- Coding questions where you already know the answer\n\n"
                "Tips:\n"
                "- Use specific, concise queries (like you'd type into Google)\n"
                "- Include the year for time-sensitive queries (e.g. 'Python 3.13 release date 2025')\n"
                "- If the first search doesn't help, try rephrasing before giving up"
            ),
            parameters=[
                ToolParameter(name="query", type=ToolParameterType.STRING,
                              description="Search query"),
                ToolParameter(name="max_results", type=ToolParameterType.INTEGER,
                              description="Maximum number of results (default 5)", required=False),
            ],
        )

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> str:
        # Try Tavily first (higher quality, AI-optimized results)
        if self._tavily_api_key:
            result = await self._search_tavily(query, max_results)
            if result is not None:
                return result
            # Tavily failed — fall through to DuckDuckGo

        return self._search_ddg(query, max_results)

    async def _search_tavily(self, query: str, max_results: int) -> str | None:
        """Search via Tavily API. Returns formatted results or None on failure."""
        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=self._tavily_api_key)
            response = await client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )

            results = []
            for r in response.get("results", []):
                score = r.get("score", 0)
                line = f"**{r['title']}** (relevance: {score:.2f})\n{r['url']}\n{r['content']}\n"
                results.append(line)

            if not results:
                return f"No results found for: {query}"
            return "\n---\n".join(results)

        except ImportError:
            logger.warning("tavily-python not installed, falling back to DuckDuckGo")
            return None
        except Exception as e:
            logger.warning("Tavily search failed, falling back to DuckDuckGo: %s", e)
            return None

    @staticmethod
    def _search_ddg(query: str, max_results: int) -> str:
        """Search via DuckDuckGo (fallback). Always returns a string."""
        try:
            from ddgs import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}\n")

            if not results:
                return f"No results found for: {query}"
            return "\n---\n".join(results)
        except ImportError:
            return "Error: No search backend available. Install tavily-python or ddgs."
        except Exception as e:
            return f"Search error: {e}"


class WebFetchTool(Tool):
    read_only = True
    def __init__(self, max_size: int = 524288, timeout: int = 15):
        self.max_size = max_size
        self.timeout = timeout

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_fetch",
            description=(
                "Fetch and extract text content from a web URL.\n\n"
                "When to use:\n"
                "- User gives you a specific URL to read\n"
                "- You found a useful link in web_search results and need the full content\n\n"
                "When NOT to use:\n"
                "- To browse or explore — use web_search to find pages first\n"
                "- For very large pages (documentation indexes, wikis) — content is truncated at 512KB\n\n"
                "Notes:\n"
                "- HTML is converted to plain text (scripts/nav/footers stripped)\n"
                "- Times out after 15 seconds — some sites may be too slow\n"
                "- Does not handle pages that require JavaScript rendering"
            ),
            parameters=[
                ToolParameter(name="url", type=ToolParameterType.STRING,
                              description="URL to fetch"),
            ],
        )

    async def execute(self, url: str, **kwargs) -> str:
        err = _is_ssrf_target(url)
        if err:
            return err

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
