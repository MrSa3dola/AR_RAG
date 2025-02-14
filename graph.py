import os

from crewai.tools import tool
from crewai_tools import RagTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

load_dotenv()

# Initialize the LLM model
PINECONE_API_KEY = os.getenv("PINE_CONE_API_KEY")
INDEX_NAME = "rag-data"

model = ChatOpenAI(model="gpt-4o")


def load_model():
    """Loads the sentence transformer model for embeddings."""
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# Define some sample tool functions for each agent
def dummy_chat_tool(message: str) -> str:
    """A simple echo tool for chat responses."""
    return f"Chat agent says: {message}"


def rag_tool(query: str) -> list:
    """
    Performs vector similarity search using Pinecone to find furniture items matching the query.

    Encodes the input query using a sentence transformer model, searches the Pinecone index,
    and returns relevant results with metadata including image paths, captions, and prices.

    Args:
        query (str): Natural language description of the desired furniture item

    Returns:
        list[dict]: List of matching items with:
            - image_path (str): URL/path to product image
            - score (float): Confidence score (0-1)
            - caption (str): Descriptive text about the item
            - price (str): Price of the item if available

    Example:
        >>> rag("modern leather sofa")
        [{'image_path': 'sofas/123.jpg', 'score': 0.87, 'caption': 'Contemporary black leather sofa...', 'price': '$799'}, ...]
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    model = load_model()
    query_vector = model.encode(query).tolist()

    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    return [
        {
            "image_path": match["metadata"].get("image_path", "N/A"),
            "score": match["score"],
            "caption": match["metadata"].get("caption", "No caption available"),
            "price": match["metadata"].get("price", "N/A"),
        }
        for match in results.get("matches", [])
    ]


def web_scrap_tool(query: str) -> list:
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
        max_results=3,
    )
    return furniture_listings


# Create the three specialized agents:
chat_agent = create_react_agent(
    model=model,
    tools=[dummy_chat_tool],
    name="chat_agent",
    prompt="You are a conversational expert. Answer naturally.",
)

rag_agent = create_react_agent(
    model=model,
    tools=[rag_tool],
    name="rag_agent",
    prompt="You are a retrieval augmented generation (RAG) expert. Use retrieval to answer questions.",
)

web_scrap_agent = create_react_agent(
    model=model,
    tools=[web_scrap_tool],
    name="web_scrap_agent",
    prompt="You are a web scraping expert. Extract and present web data clearly.",
)

# Create the supervisor that will delegate queries based on their content.
workflow = create_supervisor(
    [chat_agent, rag_agent, web_scrap_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing three agents: chat_agent, rag_agent, and web_scrap_agent. "
        "If the query is conversational or open-ended, delegate to chat_agent. "
        "If the query requires retrieval of augmented information, delegate to rag_agent. "
        "If the query asks for data that must be scraped from the web, delegate to web_scrap_agent. "
        "Decide which agent should handle the query and then output the final answer."
    ),
)

# Compile the workflow and invoke it with a user query.
app = workflow.compile()
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the latest news on OpenAI and can you scrape some data about it?",
            }
        ]
    }
)

print(result)
