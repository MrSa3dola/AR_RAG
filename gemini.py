import json

import requests

# Define the user query
user_query = "I want to recommend to me an item for a modern living room, let's say it's a black sofa"

# Make the API request
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
    },
    data=json.dumps(
        {
            "model": "google/gemini-2.0-pro-exp-02-05:free",
            "messages": [
                {
                    "role": "system",
                    "content": """Act as a system bot just extracting 
                          the furniture attributes from the input queries 
                          and make the response simple and no additional text
                          Example:
                          =====
                          user question: I want to recommend to me an item for white room, 
                          let's say it's white armchair
                          response you should return: white armchair
                          """,
                },
                {
                    "role": "user",  # Add the user query here
                    "content": user_query,
                },
            ],
        }
    ),
)

# Extract and print the assistant's response
response_data = response.json()
content = response_data["choices"][0]["message"]["content"]

print(content)
