# Add to pine_cone_tool.py or create new scraping_tool.py
import os

from crewai.tools import tool
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


@tool("web_scraper")
def web_scraper(query: str) -> list:
    """
    Searches for furniture products on Amazon Egypt using Tavily API.
    Returns list of products with details including title, price, and URL.

    Args:
        query (str): Search query (e.g., "green chaise longue")

    Returns:
        list: Formatted product results or error message
    """
    search_client = TavilyClient()

    furniture_listings = search_client.search(
        query=query + ":https://www.amazon.eg/",
        max_results=10,
    )
    return furniture_listings
