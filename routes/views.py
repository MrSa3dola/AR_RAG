import io

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import BlipForConditionalGeneration, BlipProcessor

from llm_utils import extract_features_from_caption
from recommendation import generate_caption, recommend
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
