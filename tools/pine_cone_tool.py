import os

from crewai.tools import tool
from crewai_tools import RagTool
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()
PINECONE_API_KEY = os.getenv("PINE_CONE_API_KEY")
INDEX_NAME = "rag-data"


def load_model():
    """Loads the sentence transformer model for embeddings."""
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


@tool("rag")
def rag(query: str) -> list:
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

    results = index.query(vector=query_vector, top_k=2, include_metadata=True)
    return [
        {
            "image_path": match["metadata"].get("image_path", "N/A"),
            "score": match["score"],
            "caption": match["metadata"].get("caption", "No caption available"),
            "price": match["metadata"].get("price", "N/A"),
        }
        for match in results.get("matches", [])
    ]
