import io
import os
import sys
from typing import Callable

import dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from finalp_llm import app as furniture_router
from gemini_api import extract_attributes
from read_upload import get_similar
from recommendation import (
    generate_caption,
    get_the_item_name,
    prompt_template,
    recommend,
)

# from search_agent import go_scrap

app = FastAPI()
dotenv.load_dotenv()
from fastapi.staticfiles import StaticFiles

app.include_router(furniture_router)

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

caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class MessageRequest(BaseModel):
    text: str


conversation_history = []


@app.post("/get-item")
async def get_item(request: MessageRequest):
    # print(request)

    extracted, history = extract_attributes(request.text, conversation_history)

    print("EXTRACTED == ", extracted)
    similar = get_similar(extracted)

    processed_similar = []
    for item in similar:
        image_filename = item["image_path"]
        # Extract product ID from filename (assumes format: {product_id}_image_{num}.jpg)
        product_id = image_filename.split("_image_")[0]

        # Create paths
        image_2d = f"{product_id}/{image_filename}"
        image_3d_filename = image_filename.replace(".jpg", ".glb")
        image_3d = f"{product_id}/{image_3d_filename}"
        # if item["score"] >= 0.8:
        processed_similar.append(
            {"image_2d": image_2d, "image_3d": image_3d, "score": item["score"]}
        )
    # if len(processed_similar) != 0:
    return {"content": processed_similar}
    # processed_similar = go_scrap()
    # photo = Image.open("./assets/chaise_longues_57527/chaise_longues_57527_image_2.jpg")
    # plt.imshow(photo)
    # plt.show()
    # return {"content_scrapped": processed_similar}
    return {"content_scrapped": "msh mawgoda"}


prev_items = []


@app.post("/recommend/")
async def create_upload_file(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    furniture_text = "the room contains"
    furniture_items = generate_caption(
        caption_processor, caption_model, furniture_text, image
    )

    color_text = "The wall color is"
    room_color = generate_caption(caption_processor, caption_model, color_text, image)

    recommendation = recommend(furniture_items, room_color, prev_items)

    return JSONResponse(content={"recommendation": recommendation})
