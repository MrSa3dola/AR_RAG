from crewai import Agent, Crew, Task

from agents.router_agent import (
    chat_agent,
    chat_task,
    rag_agent,
    rag_task,
    scraper_task,
    web_scraper,
)
from tools.llm import LLM
from tools.pine_cone_tool import rag
from tools.web_scraper import web_scraper


class chat:
    def __init__(self):
        self.rag_agent = rag_agent
        self.chat_agent = chat_agent
        self.web_scraper = web_scraper
    def chat(self, question):
        result = chat_agent.execute_task()
         
    def is_furniture_related(self, query: str) -> bool:
        """Check if a query is related to furniture"""
        furniture_keywords = [
            "sofa",
            "chair",
            "table",
            "desk",
            "bed",
            "furniture",
            "couch",
            "dresser",
            "cabinet",
            "shelf",
            "bookcase",
            "ottoman",
            "armchair",
            "recliner",
            "loveseat",
            "sectional",
            "nightstand",
            "wardrobe",
        ]
        return any(keyword in query.lower() for keyword in furniture_keywords)
