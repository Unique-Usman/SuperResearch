"""
Multi-source search tools for comprehensive research
Supports: Tavily, ArXiv, Wikipedia, News API (4 parallel sources)
"""
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import aiohttp
from tavily import TavilyClient
import arxiv
import wikipedia
from bs4 import BeautifulSoup
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class SearchResult:
    """Standardized search result across all sources"""
    title: str
    content: str
    url: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseSearchTool:
    """Base class for all search tools"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.source_name = "base"
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search implementation - to be overridden"""
        raise NotImplementedError
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    async def search_with_retry(self, query: str) -> List[SearchResult]:
        """Search with retry mechanism"""
        return await self.search(query)


class TavilySearch(BaseSearchTool):
    """Tavily web search"""
    
    def __init__(self, api_key: str, max_results: int = 5):
        super().__init__(max_results)
        self.client = TavilyClient(api_key=api_key)
        self.source_name = "tavily"
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Tavily"""
        try:
            # Run in executor since Tavily is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    query=query,
                    search_depth="basic",
                    max_results=self.max_results
                )
            )
            
            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    url=item.get("url", ""),
                    source="tavily",
                    score=item.get("score", 0.0),
                    metadata={"raw_content": item.get("raw_content", "")}
                ))
            
            return results
            
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []


class ArXivSearch(BaseSearchTool):
    """ArXiv academic paper search"""
    
    def __init__(self, max_results: int = 5):
        super().__init__(max_results)
        self.source_name = "arxiv"
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search ArXiv"""
        try:
            loop = asyncio.get_event_loop()
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = await loop.run_in_executor(None, lambda: list(search.results()))
            
            results = []
            for paper in papers:
                # Combine abstract and summary
                content = f"{paper.summary}\n\nAuthors: {', '.join([a.name for a in paper.authors])}"
                
                results.append(SearchResult(
                    title=paper.title,
                    content=content,
                    url=paper.entry_id,
                    source="arxiv",
                    score=1.0,  # ArXiv doesn't provide relevance scores
                    metadata={
                        "authors": [a.name for a in paper.authors],
                        "published": paper.published.isoformat(),
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url
                    }
                ))
            
            return results
            
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []


class WikipediaSearch(BaseSearchTool):
    """Wikipedia search"""
    
    def __init__(self, max_results: int = 3):
        super().__init__(max_results)
        self.source_name = "wikipedia"
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Wikipedia"""
        try:
            loop = asyncio.get_event_loop()
            
            # Search for pages
            search_results = await loop.run_in_executor(
                None,
                lambda: wikipedia.search(query, results=self.max_results)
            )
            
            results = []
            for title in search_results[:self.max_results]:
                try:
                    page = await loop.run_in_executor(None, lambda: wikipedia.page(title))
                    
                    # Get summary (first 500 chars)
                    summary = page.summary[:1000] if len(page.summary) > 1000 else page.summary
                    
                    results.append(SearchResult(
                        title=page.title,
                        content=summary,
                        url=page.url,
                        source="wikipedia",
                        score=1.0,
                        metadata={
                            "categories": page.categories[:10] if hasattr(page, 'categories') else []
                        }
                    ))
                    
                except wikipedia.exceptions.DisambiguationError:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
            
            return results
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []


class NewsAPISearch(BaseSearchTool):
    """News API for recent news articles"""
    
    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        super().__init__(max_results)
        self.api_key = api_key
        self.source_name = "news"
        self.base_url = "https://newsapi.org/v2"
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search News API"""
        if not self.api_key:
            return []
        
        try:
            async with aiohttp.ClientSession() as session:
                search_url = f"{self.base_url}/everything"
                params = {
                    "q": query,
                    "pageSize": self.max_results,
                    "sortBy": "relevancy",
                    "apiKey": self.api_key
                }
                
                async with session.get(search_url, params=params) as resp:
                    if resp.status != 200:
                        return []
                    
                    data = await resp.json()
                    articles = data.get("articles", [])
                    
                    results = []
                    for article in articles:
                        content = f"{article.get('description', '')}\n\n{article.get('content', '')}"
                        
                        results.append(SearchResult(
                            title=article.get("title", ""),
                            content=content,
                            url=article.get("url", ""),
                            source="news",
                            score=1.0,
                            metadata={
                                "published_at": article.get("publishedAt"),
                                "author": article.get("author"),
                                "source_name": article.get("source", {}).get("name")
                            }
                        ))
                    
                    return results
                    
        except Exception as e:
            print(f"News API search error: {e}")
            return []


class ParallelSearchOrchestrator:
    """Orchestrates parallel searches across multiple sources"""
    
    def __init__(
        self,
        tavily_key: str,
        news_api_key: Optional[str] = None,
        max_results_per_source: int = 5
    ):
        self.sources = {
            "tavily": TavilySearch(tavily_key, max_results_per_source),
            "arxiv": ArXivSearch(max_results_per_source),
            "wikipedia": WikipediaSearch(min(max_results_per_source, 3)),
        }
        
        if news_api_key:
            self.sources["news"] = NewsAPISearch(news_api_key, max_results_per_source)
    
    async def search_all(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        retry: bool = True
    ) -> Dict[str, List[SearchResult]]:
        """
        Search all sources in parallel
        
        Args:
            query: Search query
            sources: List of source names to search (None = all)
            retry: Whether to retry failed searches
            
        Returns:
            Dict mapping source name to list of results
        """
        if sources is None:
            sources = list(self.sources.keys())
        
        # Create tasks for parallel execution
        tasks = {}
        for source_name in sources:
            if source_name in self.sources:
                source = self.sources[source_name]
                if retry:
                    tasks[source_name] = source.search_with_retry(query)
                else:
                    tasks[source_name] = source.search(query)
        
        # Execute all searches in parallel
        results = {}
        completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for source_name, result in zip(tasks.keys(), completed):
            if isinstance(result, Exception):
                print(f"Error in {source_name}: {result}")
                results[source_name] = []
            else:
                results[source_name] = result
        
        return results
    
    def filter_empty_results(
        self,
        results: Dict[str, List[SearchResult]]
    ) -> Dict[str, List[SearchResult]]:
        """Filter out empty results"""
        return {k: v for k, v in results.items() if v}
    
    def get_all_results_flat(
        self,
        results: Dict[str, List[SearchResult]]
    ) -> List[SearchResult]:
        """Flatten results from all sources into single list"""
        flat_results = []
        for source_results in results.values():
            flat_results.extend(source_results)
        return flat_results
    
    def get_results_by_source(
        self,
        results: Dict[str, List[SearchResult]],
        source: str
    ) -> List[SearchResult]:
        """Get results from specific source"""
        return results.get(source, [])


# Utility function for easy import
async def search_all_sources(
    query: str,
    tavily_key: str,
    news_api_key: Optional[str] = None,
    max_results: int = 5,
    retry: bool = True
) -> Dict[str, List[SearchResult]]:
    """
    Convenient function to search all sources
    
    Args:
        query: Search query
        tavily_key: Tavily API key
        news_api_key: News API key (optional)
        max_results: Max results per source
        retry: Enable retry on failures
        
    Returns:
        Dict of results by source
    """
    orchestrator = ParallelSearchOrchestrator(
        tavily_key=tavily_key,
        news_api_key=news_api_key,
        max_results_per_source=max_results
    )
    
    return await orchestrator.search_all(query, retry=retry)
