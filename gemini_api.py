import os

import google.generativeai as genai
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def extract_features_from_caption(caption):
    llm = LLM(
        model="gemini/gemini-2.0-flash",  # Replace with the correct model name
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    try:
        # Define the prompt for the LLM
        prompt = f"""
        You are an AI assistant specialized in extracting key features from image captions. Your task is to analyze the given caption and return the most important features in the format: "<color> <material> <object>".

        Follow these guidelines:
        1. Identify the **most prominent color** mentioned in the caption.
           - If multiple colors are mentioned, choose the one most directly associated with the main object.
        2. Determine the **primary material** described for the main object.
           - If no specific material is mentioned, use a generic term like "material".
        3. Extract the **main object** being described.
           - Focus on the primary item in the caption.
        4. Ignore secondary details, such as background or additional objects.
        5. Return only the extracted features in the specified format.

        Example 1:
        Caption: "A red leather sofa with a wooden frame."
        Output: red wooden sofa

        Example 2:
        Caption: "The image shows a light blue chaise longue with a chaise lounger on top of it."
        Output: light blue chaise longue

        Example 3:
        Caption: "The image shows a wooden table and four chairs. The table is made of wood and the chairs are upholstered in a light-colored fabric. The chairs have a curved backrest and armrests."
        Output:  wooden table and four chairs light-colored fabric

        Example 4:
        Caption: "TThe image shows a yellow two-seater sofa bed with two pillows on top of it."
        Output: yellow sofa bed  

        Example 5:
        Caption: "The image shows a grey chair with white legs and a grey upholstered seat."
        Output: grey chair with white legs

        Example 6:
        Caption: " The image shows a bed with a blue velvet headboard and metal legs, and a white bed sheet on top. The bed is part of the Modway Furniture Collection, featuring a modern and stylish design."
        Output: bed with a blue velvet headboard and metal legs
        
        Example 7:
        Caption: " The image shows a dark grey armchair with wooden legs and a black upholstered seat."
        Output: dark grey armchair with wooden legs

        Example 8:
        Caption: " The image shows a bed with a grey headboard and wooden legs, and a white bed sheet on top. The bed frame is made of a sturdy material, and the headboard is upholstered in a neutral grey fabric."
        Output: grey bed

        Caption: {caption}

        Output:
        """
        # Assuming the LLM has a method called `call` or `process`
        result = llm.call(prompt)  # Replace `call` with the actual method name
        return result.strip()
    except Exception as e:
        print(f"Error during task execution: {e}")
        return None


# Example usage
# caption1 = "The image shows a bed with a grey upholstered headboard and metal legs, and a white mattress on top of it. The bed frame is made of a sturdy metal frame, and the headboard is upholstery"
# caption2 = "The image shows a black and white sectional sofa with a chaise lounger, perfect for relaxing and unwinding after a long day. It has a modern design with clean lines and a comfortable seating area, making it a great addition"
# caption3 = "The image shows a black recliner chair. The chair has a sleek and modern design, with a comfortable cushion and armrests. It is upholstered in a soft black fabric, giving it a luxurious look"
# caption4 = "hello, I wnat you to recommend me a beautiful sofa for my house and it's color black"

# output1 = extract_features_from_caption(caption1)
# output2 = extract_features_from_caption(caption2)
# output3 = extract_features_from_caption(caption3)
# output4 = extract_features_from_caption(caption4)

# print(f"{output1}")
# print(f"{output2}")
# print(f"{output3}")
# print(f"{output4}")
