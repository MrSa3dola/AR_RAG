import csv
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def extract_features_from_caption(caption):
    try:
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
Output: wooden table and four chairs light-colored fabric

Example 4:
Caption: "TThe image shows a yellow two-seater sofa bed with two pillows on top of it."
Output: yellow sofa bed

Example 5:
Caption: "The image shows a grey chair with white legs and a grey upholstered seat."
Output: grey chair with white legs

Example 6:
Caption: "The image shows a bed with a blue velvet headboard and metal legs, and a white bed sheet on top. The bed is part of the Modway Furniture Collection, featuring a modern and stylish design."
Output: bed with a blue velvet headboard and metal legs

Example 7:
Caption: "The image shows a dark grey armchair with wooden legs and a black upholstered seat."
Output: dark grey armchair with wooden legs

Example 8:
Caption: "The image shows a bed with a grey headboard and wooden legs, and a white bed sheet on top. The bed frame is made of a sturdy material, and the headboard is upholstered in a neutral grey fabric."
Output: grey bed

Caption: {caption}

Output:
"""
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "assistant", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during task execution: {e}")
        return None


assets_dir = "assets"  # Top-level folder

for root, dirs, files in os.walk(assets_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(root, file)
            temp_csv = csv_path + ".tmp"

            with open(csv_path, "r", newline="", encoding="utf-8") as infile, open(
                temp_csv, "w", newline="", encoding="utf-8"
            ) as outfile:

                reader = csv.DictReader(infile)
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()

                for row in reader:
                    # Modify the Caption field (adjust your edit logic as needed)
                    row["Caption"] = extract_features_from_caption(row["Caption"])
                    writer.writerow(row)

            # Replace the original CSV with the updated version
            os.replace(temp_csv, csv_path)
            print(f"Updated {csv_path}")
