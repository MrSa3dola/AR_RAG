import os

import google.generativeai as genai
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
