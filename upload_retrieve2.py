import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time

load_dotenv()

BASE_DIR = "assets2"
INDEX_NAME = "rag-data"
PINECONE_API_KEY = "pcsk_anKYK_Gz8wGefNQBjPdPoVBPmR7GAfMkUvTCgqNjv2gYVJ7op7zofvqWzVkBfSY8qzi56"
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west-2")

def get_pinecone_client():
    """Returns an instance of the Pinecone client with proper configuration."""
    return Pinecone(api_key=PINECONE_API_KEY)

def get_data():
    """Safely extracts data from all CSV files in asset folders."""
    captions, image_paths, prices = [], [], []
    
    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
            
        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue
                
            csv_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(csv_path, engine='python')
                captions.extend(df["Caption"].tolist())
                
                image_paths.extend([
                    os.path.join(folder, str(img_path)).replace('\\', '/')
                    for img_path in df["Image_Path"].tolist()
                ])
                
                if "Price" in df.columns:
                    prices.extend(df["Price"].astype(str).tolist())
                else:
                    prices.extend(["N/A"] * len(df))
                    
            except Exception as e:
                print(f"Error processing {csv_path}: {str(e)}")
                continue
                
    return captions, image_paths, prices

def load_model():
    """Loads the sentence transformer model with proper error handling."""
    try:
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def upload_to_pinecone():
    """Improved vector upload with error handling and status checks."""
    captions, image_paths, prices = get_data()
    if not captions:
        raise ValueError("No data found to upload")
        
    model = load_model()
    pc = get_pinecone_client()

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(5)

    index = pc.Index(INDEX_NAME)

    batch_size = 100
    for i in range(0, len(captions), batch_size):
        batch_captions = captions[i:i+batch_size]
        batch_images = image_paths[i:i+batch_size]
        batch_prices = prices[i:i+batch_size]
        
        embeddings = model.encode(batch_captions).tolist()
        
        vectors = [{
            "id": f"vec_{i+j}",
            "values": emb,
            "metadata": {
                "caption": cap,
                "image_path": img,
                "price": str(prc)
            }
        } for j, (cap, img, prc, emb) in enumerate(zip(
            batch_captions, batch_images, batch_prices, embeddings
        ))]
        
        try:
            index.upsert(vectors=vectors)
            print(f"Uploaded batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Failed to upload batch {i//batch_size + 1}: {str(e)}")

def get_similar(question, top_k=2):
    """Enhanced similarity search with error handling."""
    pc = get_pinecone_client()
    try:
        index = pc.Index(INDEX_NAME)
    except:
        raise ValueError("Index not found")
        
    model = load_model()
    query_vector = model.encode(question).tolist()
    
    try:
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
    except Exception as e:
        raise RuntimeError(f"Query failed: {str(e)}")
        
    return [{
        "image_path": match.metadata["image_path"],
        "score": match.score,
        "caption": match.metadata["caption"],
        "price": match.metadata.get("price", "N/A")
    } for match in results.matches]

# if __name__ == "__main__":
#     try:
#         upload_to_pinecone()
#         time.sleep(10)
#         results = get_similar("red sofa")
#         print("Search results:", results)
#     except Exception as e:
#         print(f"Error: {str(e)}")