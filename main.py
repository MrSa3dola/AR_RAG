import io
import os
import sys
from typing import Callable

import dotenv
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from gemini_api import extract_attributes
from read_upload import get_similar

# from search_agent import go_scrap

app = FastAPI()
dotenv.load_dotenv()
from fastapi.staticfiles import StaticFiles

# Mount static directory (this should be in your main FastAPI setup)
app.mount("/static", StaticFiles(directory="assets"), name="static")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class MessageRequest(BaseModel):
    text: str


@app.post("/get-item")
async def get_item(request: MessageRequest):
    # print(request)
    extracted = extract_attributes(request.text)
    print("EXTRACTED == ", extracted)
    similar = get_similar(extracted)

    processed_similar = []
    for item in similar:
        image_filename = item["image_path"]
        # Extract product ID from filename (assumes format: {product_id}_image_{num}.jpg)
        product_id = image_filename.split("_image_")[0]

        # Create paths
        image_2d = f"./assets/{product_id}/{image_filename}"
        image_3d_filename = image_filename.replace(".jpg", ".glb")
        image_3d = f"./assets/{product_id}/{image_3d_filename}"
        # if item["score"] >= 0.8:
        processed_similar.append(
            {"image_2d": image_2d, "image_3d": image_3d, "score": item["score"]}
        )
    # if len(processed_similar) != 0:
    return {"content": processed_similar[0]}
    # processed_similar = go_scrap()
    # photo = Image.open("./assets/chaise_longues_57527/chaise_longues_57527_image_2.jpg")
    # plt.imshow(photo)
    # plt.show()
    # return {"content_scrapped": processed_similar}
    return {"content_scrapped": "msh mawgoda"}
