import asyncio
import io

from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from transformers import BlipForConditionalGeneration, BlipProcessor

from llm_utils import extract_features_from_caption
from multi_agent import handle_query
from recommendation import compress_image, generate_caption, recommend
from scripts.upload_retrieve import get_similar

# Initialize router
router = APIRouter()

# Load models
caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

recommendation_history = []
conversation_history = []
prev_items = []


@router.get("/")
async def root():
    return {"message": "Hello World"}


class MessageRequest(BaseModel):
    text: str


@router.post("/get-item")
async def get_item(request: MessageRequest):
    extracted = extract_features_from_caption(request.text)

    print("EXTRACTED == ", extracted)
    similar = get_similar(extracted)

    processed_similar = []
    for item in similar:
        image_filename = item["image_path"]
        product_id = image_filename.split("_image_")[0]

        # Create paths
        image_2d = f"{product_id}/{image_filename}"
        image_3d_filename = image_filename.replace(".jpg", ".glb")
        image_3d = f"{product_id}/{image_3d_filename}"

        processed_similar.append(
            {"image_2d": image_2d, "image_3d": image_3d, "score": item["score"]}
        )

    return {"content": processed_similar}


@router.post("/recommend/")
async def create_upload_file(file: UploadFile = File(...)):
    # Read the image file into memory
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Compress the image in memory
    compressed_image = compress_image(image)
    print("compressed_image")
    # Generate captions and recommendations using the compressed image
    furniture_text = "the room contains"
    furniture_items = generate_caption(
        caption_processor, caption_model, furniture_text, compressed_image
    )

    color_text = "The wall color is"
    room_color = generate_caption(
        caption_processor, caption_model, color_text, compressed_image
    )

    recommendation = recommend(furniture_items, room_color, prev_items)
    print("recommendation =", recommendation)
    return JSONResponse(content={"recommendation": recommendation})


@router.get("/recommend/history")
async def get_recommendation_history():
    return {"history": recommendation_history}


@router.post("/chat/")
async def chat_endpoint(request: MessageRequest):
    query = request.text
    response = handle_query(query)
    return JSONResponse(content={"response": response})


@router.post("/stream/chat/")
async def stream_chat(request: MessageRequest):
    """
    Live streaming chat endpoint.
    The full response from handle_query is tokenized and streamed.
    """
    query = request.text
    full_response = handle_query(query)

    async def event_generator(response_text: str):
        # Split response into tokens (words) and stream with a slight delay
        for word in response_text.split():
            yield word + " "
            await asyncio.sleep(0.1)  # Adjust delay to control streaming pace

    return StreamingResponse(event_generator(full_response), media_type="text/plain")
