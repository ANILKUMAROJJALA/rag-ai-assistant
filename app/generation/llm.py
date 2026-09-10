import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
)


load_dotenv()


def create_llm():
    """
    Create the LLM used throughout
    the RAG application.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found."
        )

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
    )