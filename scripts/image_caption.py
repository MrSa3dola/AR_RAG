import csv
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

BASE_DIR = os.path.join("", "assets")
# print(BASE_DIR)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device = ", device)
# Load model once
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    torch_dtype=torch.float32,
    trust_remote_code=True,
).to(device)

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large", trust_remote_code=True
)


def generate_caption(image_path):
    try:
        image = Image.open(image_path)
        inputs = processor(
            text="<CAPTION>",
            images=image,
            return_tensors="pt",
        ).to(device, torch.float32)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "Caption unavailable"


def gen():
    # Process all categories
    for category in os.listdir(BASE_DIR):
        category_dir = os.path.join(BASE_DIR, category)
        csv_path = os.path.join(category_dir, "products.csv")

        if not os.path.exists(csv_path):
            continue

        # Read and update CSV
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            rows = list(csv.DictReader(csvfile))

        # Add captions
        for row in rows:
            if "Caption" not in row:
                image_path = os.path.join(category_dir, row["Image_Path"])
                if os.path.exists(image_path):
                    row["Caption"] = generate_caption(image_path)
                else:
                    row["Caption"] = "Missing image"

        # Write back to CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["ID", "Type", "Color", "Image_Path", "Caption"]
            )
            writer.writeheader()
            writer.writerows(rows)

    print("Caption generation completed!")


# gen()
print(
    generate_caption("assets\\sofas_armchairs_fu003\\sofas_armchairs_fu003_image_8.jpg")
)
