# -*- coding: utf-8 -*-
import os
import uuid

import google.generativeai as genai
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from AIAgents_final import amazon_fur

app = APIRouter()

# Configure GenAI
genai.configure(api_key=os.getenv("moaz_key"))
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
    prompt = f"""
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
    return prompt


@app.post("/start", response_model=SessionStartResponse)
async def start_session():
    session_id = str(uuid.uuid4())
    initial_message = "Hello! Please list the furniture features you are looking for. When you are finished, type 'done'."

    sessions[session_id] = {
        "conversation_history": [f"Assistant: {initial_message}"],
        "state": "collecting_features",
        "features": [],
        "retrieved": [
            "black circle expensive sofa",
            "red wood sofa",
            "black square sofa",
        ],
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
            "assistant_response": "This session has ended. Please start a new session.",
            "session_id": session_id,
        }

    current_state = session["state"]

    if current_state == "collecting_features":
        if message.lower() == "done":
            features = " ".join(session["features"])
            session["state"] = "recommending"

            response = (
                f"The best fit ones for you are {session['retrieved']}\n"
                "Based on the features you provided, would you like me to recommend one furniture option for you? Please answer yes or no."
            )
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}
        else:
            session["features"].append(message)
            prompt = "\n".join(session["conversation_history"]) + "\nAssistant:"
            llm_response = model.generate_content(prompt).text
            session["conversation_history"].append(f"Assistant: {llm_response}")
            return {"assistant_response": llm_response, "session_id": session_id}

    elif current_state == "recommending":
        if message.lower() in ["yes", "y"]:
            prompt = prompt_template(
                "Please recommend one furniture option based on my features.",
                session["retrieved"],
                " ".join(session["features"]),
            )
            recommendation = model.generate_content(prompt).text.strip()

            # Add follow-up question
            full_response = f"{recommendation}\n\nWould you like to get a new furniture recommendation? (yes/no)"

            session["conversation_history"].append(f"Assistant: {full_response}")
            session["state"] = "confirm_restart"  # Move to confirmation state
            return {"assistant_response": full_response, "session_id": session_id}

        elif message.lower() in ["no", "n"]:
            response = (
                "Alright, if you don't like these options, just tell me to get furniture from another websites store. "
                "Please type 'ok' to fetch alternative listings."
            )
            session["state"] = "awaiting_alternative"
            session["conversation_history"].append(f"Assistant: {response}")
            return {"assistant_response": response, "session_id": session_id}

        else:
            response = "Please answer with yes or no."
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
