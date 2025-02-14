import os
import uuid

import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from AIAgents_final import amazon_fur
from scripts.upload_retrieve import (
    get_similar,  # Import the vector DB retrieval function
)

app = APIRouter()

# Configure GenAI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# In-memory session storage
sessions = {}


class UserInput(BaseModel):
    session_id: str
    message: str


class SessionStartResponse(BaseModel):
    session_id: str
    assistant_response: str


def prompt_template(query, retrieved_context, features):
    return f"""
You are an expert in furniture recommendations.
Use the following retrieved context and user features to answer the question concisely (maximum three sentences).
Retrieved furniture options:
{retrieved_context}
User features: {features}
If the user's query is to receive a recommendation, please recommend one option from the retrieved items, and ask if the user needs further help.
Also, in your final sentence, include: "if you don't like this furniture just tell me to get furniture from another website furniture store."
Always end with "thanks for asking!".
Question: {query}
Helpful Answer:
"""


@app.post("/start", response_model=SessionStartResponse)
async def start_session():
    session_id = str(uuid.uuid4())
    initial_message = "Hello! Please list the furniture features you are looking for. When you are finished, type 'done'."
    sessions[session_id] = {
        "conversation_history": [f"Assistant: {initial_message}"],
        "state": "collecting_features",
        "features": [],
        "retrieved": [],
        "rag_items": [],  # This will store the exact list from get_similar()
        "active": True,
    }
    return {
        "session_id": session_id,
        "assistant_response": initial_message,
        "rag_items": sessions[session_id]["rag_items"],
    }


@app.post("/chat")
async def chat(user_input: UserInput):
    session_id = user_input.session_id
    message = user_input.message.strip()

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    session["conversation_history"].append(f"User: {message}")

    if not session["active"]:
        response = "This session has ended. Please start a new session."
        return {
            "assistant_response": response,
            "session_id": session_id,
            "rag_items": session["rag_items"],
        }

    current_state = session["state"]

    # -- State: Collecting Features --
    if current_state == "collecting_features":
        if message.lower() == "done":
            features = " ".join(session["features"])
            # Retrieve similar items from the vectorDB
            retrieved_items = get_similar(features)
            # Directly assign the returned list to session["rag_items"]
            session["rag_items"] = retrieved_items
            # Optionally format a string to display the retrieved items
            if retrieved_items:
                retrieved_str = "\n".join(
                    [
                        f"Caption: {item['caption']}, Price: {item['price']}"
                        for item in retrieved_items
                    ]
                )
            else:
                retrieved_str = "No items found."
            session["retrieved"] = retrieved_items
            session["state"] = "recommending"
            response = (
                f"Based on your features, here are the best fit options:\n{retrieved_str}\n"
                "Would you like me to recommend one furniture option for you? Please answer yes or no."
            )
            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }
        else:
            session["features"].append(message)
            prompt = "\n".join(session["conversation_history"]) + "\nAssistant:"
            llm_response = model.generate_content(prompt).text.strip()
            session["conversation_history"].append(f"Assistant: {llm_response}")
            return {
                "assistant_response": llm_response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }

    # -- State: Recommending --
    elif current_state == "recommending":
        if message.lower() in ["yes", "y"]:
            prompt = prompt_template(
                "Please recommend one furniture option based on my features.",
                session["rag_items"],
                " ".join(session["features"]),
            )
            recommendation = model.generate_content(prompt).text.strip()
            full_response = f"{recommendation}\n\nWould you like to get a new furniture recommendation? (yes/no)"
            session["conversation_history"].append(f"Assistant: {full_response}")
            session["state"] = "confirm_restart"
            return {
                "assistant_response": full_response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }
        elif message.lower() in ["no", "n"]:
            response = (
                "Alright, if you don't like these options, just tell me to get furniture from another websites store. "
                "Please type 'ok' to fetch alternative listings."
            )
            session["state"] = "awaiting_alternative"
            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }
        else:
            response = "Please answer with yes or no."
            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }

    # -- State: Awaiting Alternative --
    elif current_state == "awaiting_alternative":
        if message.lower() == "ok":
            features = " ".join(session["features"])
            alternatives = amazon_fur(features)
            alt_text = (
                alternatives.text.strip()
                if hasattr(alternatives, "text")
                else str(alternatives)
            )
            response = f"Alternative Recommendations:\n{alt_text}\nTo end the chat, please type 'thank you'."
            session["state"] = "ending"
            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }
        else:
            response = "Please type 'ok' to fetch alternative listings."
            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }

    # -- State: Ending --
    elif current_state == "ending":
        if message.lower() in ["thank", "thanks", "thank you"]:
            response = (
                "You're welcome! It was my pleasure helping you.\n"
                "Would you like to get a new furniture recommendation? (yes/no)"
            )
            session["state"] = "confirm_restart"
            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }
        else:
            prompt = "\n".join(session["conversation_history"]) + "\nAssistant:"
            llm_response = model.generate_content(prompt).text.strip()
            session["conversation_history"].append(f"Assistant: {llm_response}")
            return {
                "assistant_response": llm_response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }

    # -- State: Confirm Restart --
    elif current_state == "confirm_restart":
        if message.lower() in ["yes", "y"]:
            initial_message = "Great! Let's start over. Please list the furniture features you're looking for. Type 'done' when finished."
            session.update(
                {
                    "conversation_history": [f"Assistant: {initial_message}"],
                    "state": "collecting_features",
                    "features": [],
                    "rag_items": [],
                    "active": True,
                }
            )
            return {
                "assistant_response": initial_message,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }
        elif message.lower() in ["no", "n"]:
            goodbye_message = "Thank you for using our service! Have a great day!"
            session["active"] = False
            session["conversation_history"].append(f"Assistant: {goodbye_message}")
            return {
                "assistant_response": goodbye_message,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }
        else:
            response = "Please answer with yes or no."
            session["conversation_history"].append(f"Assistant: {response}")
            return {
                "assistant_response": response,
                "session_id": session_id,
                "rag_items": session["rag_items"],
            }

    # -- Fallback --
    response = "I'm sorry, I didn't understand that."
    session["conversation_history"].append(f"Assistant: {response}")
    return {
        "assistant_response": response,
        "session_id": session_id,
        "rag_items": session["rag_items"],
    }
# hello, 
# 
# 