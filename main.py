import io
import os
import sys
from typing import Callable

import dotenv
from fastapi import FastAPI, File, UploadFile  # Changed import
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from gemini_api import extract_attributes
from read_upload import get_similar

app = FastAPI()
dotenv.load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/get-item")
async def get_item(file: UploadFile = File(...)):  # Changed parameter
    # Read and decode the file
    contents = await file.read()
    text = contents.decode("utf-8")

    extracted = extract_attributes(text)
    similar = get_similar(extracted)
    return similar
