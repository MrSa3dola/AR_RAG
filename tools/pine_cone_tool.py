import os

from crewai.tools import tool
from crewai_tools import RagTool
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


def load_model():
    from sentence_transformers import SentenceTransformer

    # Load the model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model


@tool("rag")
def rag(query: str) -> str:
    """
    Performs vector similarity search using Pinecone to find furniture items matching the query.

    Encodes the input query using a sentence transformer model, searches the Pinecone index,
    and returns relevant results with metadata including image paths and captions.

    Args:
        query (str): Natural language description of the desired furniture item

    Returns:
        list[dict]: List of matching items with:
            - image_path (str): URL/path to product image
            - score (float): Confidence score (0-1)
            - caption (str): Descriptive text about the item

    Example:
        >>> rag("modern leather sofa")
        [{'image_path': 'sofas/123.jpg', 'score': 0.87, 'caption': 'Contemporary black leather sofa...'}, ...]
    """

    pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))

    index = pc.Index("ar-rag")
    model = load_model()
    embedding = model.encode(query)
    data = index.query(
        vector=embedding.tolist(),
        top_k=2,
        include_metadata=True,
    )
    result = [
        {
            "image_path": match["metadata"]["image_path"],
            "score": match["score"],
            "caption": match["metadata"]["caption"],
        }
        for match in data["matches"]
    ]
    return result
