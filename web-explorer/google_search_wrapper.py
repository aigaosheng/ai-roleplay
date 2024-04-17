"""Util that calls DuckDuckGo Search.

No setup required. Free.
https://pypi.org/project/google/
"""
from typing import Dict, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
import googlesearch

class GoogleSearchAPIWrapperLocal(BaseModel):
    """Wrapper for DuckDuckGo Search API.

    Free and does not require any setup.
    """

    region: Optional[str] = "wt-wt"
    safesearch: str = "moderate"
    time: Optional[str] = "y"
    max_results: int = 5

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        try:
            import googlesearch  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import googlesearch python package. "
                "Please install it with `pip install -U google`."
            )
        return values

    def _ddgs_text(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo text search and return results."""
        # from duckduckgo_search import DDGS

        ddgs_gen = googlesearch.search(
            query,
            # region=self.region,
            # safesearch=self.safesearch,
            # timelimit=self.time,
            stop=max_results or self.max_results,
        )
        if ddgs_gen:
            return [r for r in ddgs_gen]
        return []

    def run(self, query: str) -> str:
        """Run query through DuckDuckGo and return concatenated results."""
        results = self._ddgs_text(query)

        if not results:
            return "No good google Search Result was found"
        try:
            return " ".join(r["body"] for r in results)
        except:
            return ""

    def results(
        self, query: str, max_results: int
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo and return metadata.

        Args:
            query: The query to search for.
            max_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        results = [
            {"snippet": "", "title": "", "link": r}
            for r in self._ddgs_text(query, max_results=max_results)
        ]

        if results is None:
            results = [{"Result": "No good Google Search Result was found"}]

        return results
