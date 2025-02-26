import os

from langchain_openai import ChatOpenAI


class LLM:
    @staticmethod
    def llm(temperature=0.5, max_tokens=1000):
        """Returns configured Gemini LLM instance"""
        api_key = os.getenv("GEMENI_API_KEY_2")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable")

        return ChatOpenAI(
            model_name="gemini/gemini-2.0-flash",
            temperature=temperature,
            openai_api_key=api_key,
            max_tokens=max_tokens,
        )
