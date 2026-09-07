import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def create_llm():
    """Create the OpenAI chat model."""

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-5.6")

    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0,
    )


if __name__ == "__main__":
    llm = create_llm()

    response = llm.invoke(
        "Explain what Retrieval-Augmented Generation (RAG) is in one sentence."
    )

    print(response.content)