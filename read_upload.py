import os

from dotenv import load_dotenv

load_dotenv()
import pandas as pd
import pinecone
from pinecone import Pinecone


def get_captions():
    # Define the base directory
    base_dir = "assets"

    # Initialize a list to store all captions
    captions = []
    image_paths = []
    # Iterate through each folder
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            # Look for the CSV file in the folder
            csv_file = None
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    csv_file = os.path.join(folder_path, file)
                    break

            if csv_file:
                # Load the CSV file
                df = pd.read_csv(csv_file)
                # Extract the 'Caption' column and add to the list
                captions.extend(df["Caption"].tolist())
                image_paths.extend(df["Image_Path"].tolist())
    return captions, image_paths


# Now `captions` contains all the captions from all CSV files


def load_model():
    from sentence_transformers import SentenceTransformer

    # Load the model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model


def upload():
    # Generate embeddings
    captions, image_paths = get_captions()
    print("Captions check")
    print(captions[0])
    print("-------------------")
    model = load_model()
    embeddings = model.encode(captions)
    print("Embedding shape")
    print(embeddings.shape)
    # Initialize Pinecone

    pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))
    index = pc.Index("ar-rag")
    print("PineCone attached")
    # Prepare data for upload (including image_path)
    ids = [f"vec_{i}" for i in range(len(captions))]
    metadata = [
        {"caption": caption, "image_path": image_path}
        for caption, image_path in zip(captions, image_paths)
    ]

    # Upload to Pinecone
    index.upsert(vectors=zip(ids, embeddings.tolist(), metadata))

    print("Embeddings uploaded to Pinecone.")

    # Query the index
    query_vector = embeddings[0].tolist()  # Use the first embedding as a query
    query_results = index.query([query_vector], top_k=5, include_metadata=True)

    # Print results with image path
    for match in query_results["matches"]:
        print(
            f"ID: {match['id']}, Score: {match['score']}, Caption: {match['metadata']['caption']}, Image Path: {match['metadata']['image_path']}"
        )


# upload()
def get_similar(question):
    pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))
    index = pc.Index("ar-rag")
    model = load_model()
    embedding = model.encode(question)
    data = index.query(
        vector=embedding.tolist(),
        top_k=4,
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


# print(get_similar("gray wardrobe"))
