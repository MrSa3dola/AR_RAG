# # import json
# # import os

# # import requests
# # from dotenv import load_dotenv

# # load_dotenv()


# # def extract_attributes(user_query):
# #     # Make the API request
# #     response = requests.post(
# #         url="https://openrouter.ai/api/v1/chat/completions",
# #         headers={
# #             "Authorization": "Bearer " + os.getenv("OPEN_ROUTER_API"),
# #         },
# #         data=json.dumps(
# #             {
# #                 "model": "google/gemini-2.0-pro-exp-02-05:free",
# #                 "messages": [
# #                     {
# #                         "role": "system",
# #                         "content": """Act as a system bot just extracting
# #                             the furniture attributes from the input queries
# #                             and make the response simple and no additional text
# #                             Example:
# #                             =====
# #                             user question: I want to recommend to me an item for white room,
# #                             let's say it's white armchair
# #                             response you should return: white armchair
# #                             """,
# #                     },
# #                     {
# #                         "role": "user",  # Add the user query here
# #                         "content": user_query,
# #                     },
# #                 ],
# #             }
# #         ),
# #     )

# #     # Extract and print the assistant's response
# #     response_data = response.json()
# #     content = response_data["choices"][0]["message"]["content"]

# import os

# #     return content
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# def extract_attributes(user_query):
#     model = genai.GenerativeModel("gemini-pro")
#     response = model.generate_content(
#         [
#             {"role": "user", "parts": [
#                 "Extract only the furniture attributes from the following query. "
#                 "Do not include any explanations or additional text. Just return "
#                 "the attributes as a concise phrase or list.\n\n"
#                 "Example:\n"
#                 "User query: I want a modern white armchair for my minimalist room.\n"
#                 "Response: modern white armchair\n\n"
#                 f"User query: {user_query}"
#             ]}
#         ]
#     )

#     return response.text.strip()

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def extract_attributes(user_query):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        [
            {
                "role": "user",
                "parts": [
                    "Extract only the furniture attributes from the following query. "
                    "Do not include any explanations or additional text. Just return "
                    "the attributes as a concise phrase or list.\n\n"
                    "Example:\n"
                    "User query: I want a modern white armchair for my minimalist room.\n"
                    "Response: modern white armchair\n\n"
                    f"User query: {user_query}"
                ],
            }
        ]
    )
    response2 = model.generate_content(
        [
            {
                "role": "model",
                "parts": [
                    "You are a specialized AI that extracts only the furniture attributes "
                    "from the user's input. Your response should be a concise phrase or list "
                    "without explanations or additional text."
                ],
            },
            {
                "role": "user",
                "parts": [
                    "Example:\n"
                    "User query: I want a modern white armchair for my minimalist room.\n"
                    "Response: modern white armchair\n\n"
                    f"User query: {user_query}"
                ],
            },
        ]
    )
    return response2.text.strip()


# a, b = extract_attributes("recommend to me a white sofa and I want it to be modern")
# print(a)
# print(b)
