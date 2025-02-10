import io
import os
import sys
from typing import Callable

import dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from gemini_api import extract_attributes
from read_upload import get_similar

app = FastAPI()
dotenv.load_dotenv()
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

    # extract from gemini
    extracted = extract_attributes(request.text)
    # similarity search from pince cone
    similar = get_similar(extracted)
    # similar -> list of {image_path_2d, image_path_3d, score}
    return {"content": similar[0]}
    # image_2d = get_image_2d(image_path_2d)
    # image_3d = get_image_3d(image_path_3d)
    # return image_3d
