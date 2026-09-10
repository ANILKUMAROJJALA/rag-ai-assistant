from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------
# Chat request
# --------------------------------------------------

class ChatRequest(BaseModel):
    """
    Request body sent to the RAG chat endpoint.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG assistant.",
    )

    thread_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Conversation identifier used by "
            "LangGraph checkpointing."
        ),
    )


# --------------------------------------------------
# Source information
# --------------------------------------------------

class SourceInfo(BaseModel):
    """
    Metadata describing a source used
    to generate the answer.
    """

    source: Optional[str] = None

    file_type: Optional[str] = None

    chunk_id: Optional[str] = None

    page: Optional[str] = None


# --------------------------------------------------
# Chat response
# --------------------------------------------------

class ChatResponse(BaseModel):
    """
    Structured response returned by
    the RAG chat endpoint.
    """

    question: str

    answer: str

    thread_id: str

    route: str

    retrieval_relevant: Optional[bool] = None

    standalone_question: Optional[str] = None

    sources: list[SourceInfo] = []