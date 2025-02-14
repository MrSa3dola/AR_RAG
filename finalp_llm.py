import os
import uuid

import google.generativeai as genai
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from AIAgents_final import amazon_fur
from scripts.upload_retrieve import get_similar
from upload_retrieve2 import get_similar

app = APIRouter()

# Configure GenAI
token = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=token)
model = genai.GenerativeModel("gemini-1.5-flash")

# Session storage
sessions = {}


class UserInput(BaseModel):
    session_id: str
    message: str


class SessionStartResponse(BaseModel):
    session_id: str
    assistant_response: str


def prompt_template(query, retrieved_context, features):
    formatted_context = "\n".join(
        [
            f"- {item['caption']} (Price: {item['price']}, Relevance: {item['score']:.2f})"
            for item in retrieved_context
        ]
    )

    prompt = f"""
You are an expert in furniture recommendations. Use the following context and user features to answer concisely (max 3 sentences).

Retrieved options (sorted by relevance):
{formatted_context}

User preferences: {features}

When recommending:
1. Choose the best match considering price and relevance score
2. Mention price but not technical scores
3. Include: "Don't like this? Ask for other stores!"
4. Always end with "Thanks for asking!"

Question: {query}
Helpful Answer:
"""
    return prompt


@app.post("/start", response_model=SessionStartResponse)
async def start_session():
    session_id = str(uuid.uuid4())
    initial_message = "Hello! Please list furniture features you're looking for. Type 'done' when finished."

    sessions[session_id] = {
        "conversation_history": [f"Assistant: {initial_message}"],
        "state": "collecting_features",
        "features": [],
        "retrieved": [],
        "image_paths": [],
        "active": True,
    }

    return {"session_id": session_id, "assistant_response": initial_message}


@app.post("/chat")
async def chat(user_input: UserInput):
    session_id = user_input.session_id
    message = user_input.message.strip()

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    session["conversation_history"].append(f"User: {message}")

    if not session["active"]:
        return {
            "assistant_response": "Session ended. Start new session.",
            "session_id": session_id,
        }

    current_state = session["state"]

    if current_state == "collecting_features":
        if message.lower() == "done":
            features = " ".join(session["features"])

            try:
                similar_items = get_similar(features)
                session["retrieved"] = [
                    {
                        "caption": item["caption"],
                        "price": item.get("price", "N/A"),
                        "score": item["score"],
                    }
                    for item in similar_items
                ]
                session["image_paths"] = [item["image_path"] for item in similar_items]
            except Exception as e:
                print(f"Retrieval error: {e}")
                session["retrieved"] = []
                session["image_paths"] = []

            session["state"] = "recommending"

            response_items = "\n".join(
                [
                    f"- {item['caption']} (Price: {item['price']})"
                    for item in session["retrieved"]
                ]
            )
            response = f"Top matches:\n{response_items}\n\nRecommend one? (yes/no)"

            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "internal_data": {"images": session["image_paths"]},  # Hidden from user
            }
        else:
            session["features"].append(message)
            prompt = "\n".join(session["conversation_history"]) + "\nAssistant:"
            llm_response = model.generate_content(prompt).text
            session["conversation_history"].append(f"Assistant: {llm_response}")
            return {"assistant_response": llm_response, "session_id": session_id}

    elif current_state == "recommending":
        if message.lower() in ["yes", "y"]:
            prompt = prompt_template(
                "Recommend one option based on my features.",
                session["retrieved"],
                " ".join(session["features"]),
            )
            recommendation = model.generate_content(prompt).text.strip()

            full_response = f"{recommendation}\n\nNew recommendation? (yes/no)"

            session["conversation_history"].append(f"Assistant: {full_response}")
            session["state"] = "confirm_restart"
            return {
                "assistant_response": full_response,
                "session_id": session_id,
                "internal_data": {"images": session["image_paths"]},
            }

        elif message.lower() in ["no", "n"]:
            response = "Don't like these? Type 'ok' for other stores."
            session["state"] = "awaiting_alternative"
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}

        else:
            response = "Please answer yes/no."
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}

    elif current_state == "awaiting_alternative":
        if message.lower() == "ok":
            features = " ".join(session["features"])
            alternatives = amazon_fur(features)
            alt_text = (
                alternatives.text
                if hasattr(alternatives, "text")
                else str(alternatives)
            )
            response = f"Alternative Recommendations:\n{alt_text}\nTo end the chat, please type 'thank you'."
            session["state"] = "ending"
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}
        else:
            response = "Please type 'ok' to fetch alternative listings."
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}

    elif current_state == "ending":
        if message.lower() in ["thank", "thanks", "thank you"]:
            response = "You're welcome! It was my pleasure helping you.\nWould you like to get a new furniture recommendation? (yes/no)"
            session["state"] = "confirm_restart"
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}
        else:
            prompt = "\n".join(session["conversation_history"]) + "\nAssistant:"
            llm_response = model.generate_content(prompt).text
            session["conversation_history"].append(f"Assistant: {llm_response}")
            return {"assistant_response": llm_response, "session_id": session_id}

    elif current_state == "confirm_restart":
        if message.lower() in ["yes", "y"]:
            initial_message = "Great! Let's start over. Please list the furniture features you're looking for. Type 'done' when finished."
            session.update(
                {
                    "conversation_history": [f"Assistant: {initial_message}"],
                    "state": "collecting_features",
                    "features": [],
                    "active": True,
                }
            )
            return {"assistant_response": initial_message, "session_id": session_id}
        elif message.lower() in ["no", "n"]:
            goodbye_message = "Thank you for using our service! Have a great day!"
            session["active"] = False
            session["conversation_history"].append(f"Assistant: {goodbye_message}")
            return {"assistant_response": goodbye_message, "session_id": session_id}
        else:
            response = "Please answer with yes or no."
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}
