from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)


class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
    )

    thread_id: str = Field(
        ...,
        min_length=1,
    )


class SourceInfo(BaseModel):

    source: Optional[
        str
    ] = None

    file_type: Optional[
        str
    ] = None

    chunk_id: Optional[
        str
    ] = None

    page: Optional[
        str
    ] = None


class ChatResponse(BaseModel):

    question: str

    answer: str

    thread_id: str

    route: str

    retrieval_relevant: Optional[
        bool
    ] = None

    standalone_question: Optional[
        str
    ] = None

    sources: list[
        SourceInfo
    ] = Field(
        default_factory=list
    )


class ConversationSummary(
    BaseModel
):

    id: str

    title: str

    created_at: str

    updated_at: str


class ConversationMessage(
    BaseModel
):

    id: str

    role: str

    content: str

    sources: list[
        SourceInfo
    ] = Field(
        default_factory=list
    )

    route: Optional[
        str
    ] = None

    retrieval_relevant: Optional[
        bool
    ] = None

    created_at: str


class ConversationDetail(
    ConversationSummary
):

    messages: list[
        ConversationMessage
    ] = Field(
        default_factory=list
    )


class DocumentInfo(
    BaseModel
):

    name: str

    file_type: str

    size: int

    status: str

    chunks: Optional[
        int
    ] = None