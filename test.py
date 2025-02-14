import json
import re

from crewai import Agent, Crew, Process, Task

from tools.llm import LLM
from tools.pine_cone_tool import rag
from tools.web_scraper import web_scraper

# Define agents with improved prompts and backstories
chat_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="chat_agent",
    goal="Handle general conversation and provide helpful responses to non-furniture queries",
    backstory="I am a conversational assistant trained to engage users in friendly dialogue, answer general questions, and provide information on a wide range of topics outside of furniture recommendations.",
    allow_delegation=False,
    verbose=True,
)

rag_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="rag_agent",
    goal="Retrieve and recommend relevant furniture items using vector search",
    backstory="I am a specialized retrieval agent with access to a furniture database. I can find items that match specific requirements and provide detailed information about them, along with confidence scores for my recommendations.",
    tools=[rag],
    allow_delegation=False,
    verbose=True,
)

scrap_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="scrap_agent",
    goal="Find and analyze furniture listings from online sources to match user requirements",
    backstory="I am an expert at finding furniture products online, extracting their specifications, and matching them against user requirements. I can provide detailed product information including dimensions, materials, and design features.",
    tools=[web_scraper],
    allow_delegation=False,
    verbose=True,
)

# Define tasks with JSON output structure
chat_task = Task(
    description="""
    Respond to the user query: {query}
    
    Format your response as a JSON string with the following structure:
    {
        "response": "Your friendly conversation response",
        "follow_up_questions": ["optional question 1", "optional question 2"]
    }
    """,
    agent=chat_agent,
    expected_output="A JSON-formatted friendly conversation response",
)

rag_task = Task(
    description="""
    Find furniture matching the query: {query}
    
    Format your response as a JSON string with the following structure:
    {
        "results": [
            {
                "item_name": "Name of furniture item",
                "description": "Brief description",
                "match_score": 0.95,
                "specifications": {"material": "...", "dimensions": "..."}
            }
        ],
        "confidence": 0.85
    }
    """,
    agent=rag_agent,
    expected_output="A JSON-formatted list of furniture recommendations with confidence scores",
)

scraper_task = Task(
    description="""
    Search for furniture products matching: {query}
    
    Extract specific details including dimensions, materials, design features, and price.
    Compare these against the user's requirements.
    
    Format your response as a JSON string with the following structure:
    {
        "results": [
            {
                "product_name": "Name of product",
                "url": "Product URL",
                "price": "$X.XX",
                "specifications": {"dimensions": "...", "material": "..."},
                "match_reasoning": "Why this matches the user query"
            }
        ],
        "search_parameters": "What was searched for"
    }
    """,
    agent=scrap_agent,
    expected_output="A JSON-formatted set of best-matching furniture products from web search",
)

# Router agent with delegation fix
router_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="router_agent",
    goal="Analyze user queries and direct them to the most appropriate specialized agent",
    backstory="I am the central coordination point for all user interactions. I understand query intent and context to determine whether a request should be handled by the furniture recommendation systems or general conversation.",
    verbose=True,
    allow_delegation=True,
)

router_task = Task(
    description="""
    Analyze the following query and determine which agent should handle it: {query}
    
    Step 1: Categorize the query as either furniture-related or general conversation.
    Step 2: For furniture queries, decide whether it needs vector search (rag_agent) or web scraping (scrap_agent).
    Step 3: Return your decision as a JSON structure.
    
    Format your response as a JSON string:
    {
        "selected_agent": "chat_agent|rag_agent|scrap_agent",
        "reasoning": "Brief explanation for this selection",
        "query_category": "furniture|general|other"
    }
    """,
    agent=router_agent,
    expected_output="A JSON-formatted decision about which agent should handle the query",
)
crew = Crew(
    # manager_llm=LLM.llm(temperature=0),
    manager_agent=router_agent,
    agents=[chat_agent, rag_agent, scrap_agent],
    tasks=[chat_task, rag_task, scraper_task],
    # memory=True,
    process=Process.hierarchical,
)
query = "hello"
inputs = {"query": query}
result = crew.kickoff(inputs=inputs)
print(result)
