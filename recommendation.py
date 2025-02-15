import io
import os

import google.generativeai as genai
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Initialize the models and configurations

token = os.getenv("GEMENI_API_KEY_2")
genai.configure(api_key=token)
llm_model = genai.GenerativeModel("gemini-1.5-flash")


def generate_caption(caption_processor, caption_model, text, raw_image):
    inputs = caption_processor(raw_image, text, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)


def prompt_template(furniture_items, room_color, prev_items):
    prompt = f"""
You are an experienced interior designer.
Based on the following retrieved furniture items, the room color, and the given color palette, recommend another item that will fit the style well.
Retrieved furniture items:
{furniture_items}
room color: {room_color}
Recommend the furniture item including the item shape and item color.
Do not recommend an item that is in the furniture items.
Do not recommend a furniture item that has been suggested before.
Previously recommended: {prev_items}.
recommended item:
"""
    return prompt


def get_the_item_name(item):
    prompt = f"extract the furniture item name only from the following text: {item}"
    response = llm_model.generate_content(prompt)
    return response.text.strip()


def recommend(furniture_items, room_color, prev_items):
    prompt = prompt_template(furniture_items, room_color, prev_items)
    response = llm_model.generate_content(prompt)
    recommendation = response.text.strip()
    prev_items.append(get_the_item_name(recommendation))
    return recommendation


def compress_image(image, max_size_mb=1, quality=85):

    # Convert image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image to reduce dimensions
    max_dimension = 1024  # Adjust this value as needed
    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    # Compress the image by reducing quality
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)

    # Check the size of the compressed image
    while output.tell() > max_size_mb * 1024 * 1024 and quality > 10:
        quality -= 5
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality)

    # Reset the stream position to the beginning
    output.seek(0)

    # Return the compressed image as a PIL.Image object
    return Image.open(output)
