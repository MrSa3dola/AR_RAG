import csv
import json
import os

import requests
from bs4 import BeautifulSoup

urls = [
    "https://www.ikea.com/eg/en/cat/chaise-longues-57527/",
    "https://www.ikea.com/eg/en/cat/wardrobes-19053/",
    "https://www.ikea.com/eg/en/cat/sofas-armchairs-fu003/",
    "https://www.ikea.com/eg/en/cat/upholstered-beds-49096/",
    "https://www.ikea.com/eg/en/cat/armchairs-16239/",
    "https://www.ikea.com/eg/en/cat/upholstered-chairs-25221/",
    "https://www.ikea.com/eg/en/cat/dining-sets-19145/",
    "https://www.ikea.com/eg/en/cat/sideboards-buffets-console-tables-30454/",
    "https://www.ikea.com/eg/en/cat/cabinets-cupboards-st003/",
    "https://www.ikea.com/eg/en/cat/tv-media-furniture-10475/",
]
BASE_DIR = os.path.join("AR_RAG", "assets")
os.makedirs(BASE_DIR, exist_ok=True)

for url in urls:
    # Create category folder
    category = url.split("/cat/")[1].split("/")[0].replace("-", "_")
    category_dir = os.path.join(BASE_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    # Scrape and download images
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Create CSV in category folder
    csv_path = os.path.join(category_dir, "products.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Type", "Color", "Image_Path"])

        image_div = soup.find("div", class_="plp-catalog-product-list")
        if image_div:
            img_tags = image_div.find_all("img", class_="plp-image plp-product__image")
            for i, img in enumerate(img_tags[:10]):
                if not img.get("src"):
                    continue

                # Download image
                img_url = img["src"]
                img_response = requests.get(img_url, stream=True)
                if img_response.status_code != 200:
                    continue

                # Save image
                filename = f"{category}_image_{i+1}.jpg"
                filepath = os.path.join(category_dir, filename)
                with open(filepath, "wb") as f:
                    for chunk in img_response.iter_content(1024):
                        f.write(chunk)

                # Write to CSV
                writer.writerow(
                    [
                        i + 1,
                        category.replace("_", " ").title(),
                        "N/A",  # Add color extraction logic
                        filename,
                    ]
                )
