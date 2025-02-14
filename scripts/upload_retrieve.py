import os

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = "assets"
INDEX_NAME = "rag-data"
PINECONE_API_KEY = os.getenv("PINE_CONE_API_KEY")


def get_data():
    """Extracts captions, image paths, and prices from CSV files in asset folders."""
    captions, image_paths, prices = [], [], []
    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.isdir(folder_path):
            csv_file = next(
                (
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.endswith(".csv")
                ),
                None,
            )
            if csv_file:
                df = pd.read_csv(csv_file)
                captions.extend(df["Caption"].tolist())
                image_paths.extend(df["Image_Path"].tolist())
                prices.extend(
                    df.get("Price", [None] * len(df))
                )  # Handle missing price column
    return captions, image_paths, prices


# Now `captions` contains all the captions from all CSV files


def load_model():
    """Loads the sentence transformer model for embeddings."""
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def upload_to_pinecone():
    """Uploads embeddings along with captions, image paths, and prices to Pinecone."""
    captions, image_paths, prices = get_data()
    model = load_model()
    embeddings = model.encode(captions).tolist()

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Prepare data for upload
    ids = [f"vec_{i}" for i in range(len(captions))]
    metadata = [
        {"caption": caption, "image_path": image_path, "price": price}
        for caption, image_path, price in zip(captions, image_paths, prices)
    ]

    # Upload to Pinecone
    index.upsert(vectors=zip(ids, embeddings, metadata))
    print("Embeddings uploaded to Pinecone.")


def get_similar(question, top_k=2):
    """Retrieves the top_k most similar items based on the query."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    model = load_model()
    query_vector = model.encode(question).tolist()

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [
        {
            "image_path": match["metadata"]["image_path"],
            "score": match["score"],
            "caption": match["metadata"]["caption"],
            "price": match["metadata"].get("price", "N/A"),
        }
        for match in results["matches"]
    ]


# upload_to_pinecone()
# print(get_similar("light blue sofa"))
