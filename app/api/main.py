import logging

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.api.documents import (
    delete_document,
    list_documents,
    save_uploaded_file,
)

from app.api.history import (
    create_conversation,
    delete_conversation,
    ensure_conversation,
    get_conversation,
    get_conversations,
    initialize_history_database,
    maybe_create_title,
    save_message,
)

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    DocumentInfo,
)

from app.config import (
    CORS_ORIGINS,
)


logger = logging.getLogger(
    "uvicorn.error"
)


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize application resources
    when the FastAPI application starts.

    Code before yield runs during startup.
    Code after yield would run during shutdown.
    """
    initialize_history_database()

    yield


app = FastAPI(
    title="RAG AI Assistant API",
    description=(
        "API for the production-style "
        "RAG AI Assistant."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=CORS_ORIGINS,

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


@lru_cache(maxsize=1)
def get_rag_app():
    """
    Lazily build and cache the
    LangGraph RAG application.
    """
    from app.generation.graph import (
        build_graph,
    )

    return build_graph()


def create_initial_state(
    question: str,
):
    """
    Create the full initial LangGraph
    state for a brand-new thread.
    """
    return {
        "question": question,
        "standalone_question": "",
        "conversation_history": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "reranker_scores": [],
        "answer": "",
        "sources": [],
        "route": "",
        "metadata_filter": {},
        "retrieval_relevant": None,
    }


@app.get("/")
def root():
    return {
        "message":
            "RAG AI Assistant API",

        "docs":
            "/docs",

        "health":
            "/health",

        "chat":
            "/chat",

        "conversations":
            "/conversations",

        "documents":
            "/documents",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service":
            "RAG AI Assistant",
    }


# --------------------------------------------------
# Conversations
# --------------------------------------------------


@app.post(
    "/conversations",
    response_model=ConversationSummary,
)
def create_new_conversation():
    """
    Create a brand-new conversation
    for the frontend New Chat action.
    """
    return create_conversation()


@app.get(
    "/conversations",
    response_model=list[
        ConversationSummary
    ],
)
def list_conversations():
    """
    Return all persisted conversations.
    """
    return get_conversations()


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
)
def read_conversation(
    conversation_id: str,
):
    """
    Return one conversation together
    with its complete visible history.
    """
    conversation = (
        get_conversation(
            conversation_id
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found."
            ),
        )

    return conversation


@app.delete(
    "/conversations/{conversation_id}"
)
def remove_conversation(
    conversation_id: str,
):
    """
    Delete a persisted conversation
    and its visible messages.
    """
    deleted = delete_conversation(
        conversation_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found."
            ),
        )

    return {
        "deleted": True,
        "conversation_id":
            conversation_id,
    }


# --------------------------------------------------
# Documents
# --------------------------------------------------


@app.get(
    "/documents",
    response_model=list[
        DocumentInfo
    ],
)
def read_documents():
    """
    Return documents currently available
    to the application's knowledge base.
    """
    return list_documents()


@app.post(
    "/documents/upload",
    response_model=DocumentInfo,
)
async def upload_document(
    file: UploadFile = File(
        ...
    ),
):
    """
    Upload and index a supported
    knowledge-base document.
    """
    logger.info(
        "Document upload received: %s",
        file.filename,
    )

    result = (
        await save_uploaded_file(
            file
        )
    )

    logger.info(
        (
            "Document indexed: "
            "name=%s chunks=%s"
        ),
        result["name"],
        result.get(
            "chunks"
        ),
    )

    return result


@app.delete(
    "/documents/{filename:path}"
)
def remove_document(
    filename: str,
):
    """
    Remove a document from storage
    and the vector index.
    """
    logger.info(
        "Document delete requested: %s",
        filename,
    )

    return delete_document(
        filename
    )


# --------------------------------------------------
# Chat
# --------------------------------------------------


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    rag_app=Depends(
        get_rag_app
    ),
):
    """
    Process one conversational turn.

    If the supplied thread_id does not yet
    exist in the application history database,
    it is automatically created.

    This keeps /chat backward compatible while
    still supporting the frontend's explicit
    New Chat workflow.
    """
    logger.info(
        (
            "Chat request received: "
            "thread_id=%s"
        ),
        request.thread_id,
    )

    try:
        # --------------------------------------------------
        # Ensure application-level conversation exists
        # --------------------------------------------------

        ensure_conversation(
            request.thread_id
        )

        # --------------------------------------------------
        # Save visible user message
        # --------------------------------------------------

        save_message(
            conversation_id=(
                request.thread_id
            ),
            role="user",
            content=request.question,
        )

        # --------------------------------------------------
        # Automatically create sidebar title
        # from the first user question
        # --------------------------------------------------

        maybe_create_title(
            request.thread_id,
            request.question,
        )

        # --------------------------------------------------
        # LangGraph thread configuration
        # --------------------------------------------------

        config = {
            "configurable": {
                "thread_id":
                    request.thread_id
            }
        }

        # --------------------------------------------------
        # Check whether LangGraph already has
        # checkpointed state for this thread
        # --------------------------------------------------

        snapshot = (
            rag_app.get_state(
                config
            )
        )

        existing_state = (
            snapshot.values
            if snapshot
            else {}
        )

        # --------------------------------------------------
        # First turn needs complete initial state.
        # Follow-up turns only need the new question
        # because LangGraph restores checkpointed state.
        # --------------------------------------------------

        if not existing_state:

            graph_input = (
                create_initial_state(
                    request.question
                )
            )

        else:

            graph_input = {
                "question":
                    request.question
            }

        # --------------------------------------------------
        # Run LangGraph RAG workflow
        # --------------------------------------------------

        result = rag_app.invoke(
            graph_input,
            config=config,
        )

        sources = result.get(
            "sources",
            [],
        )

        # --------------------------------------------------
        # Save visible assistant response
        # --------------------------------------------------

        save_message(
            conversation_id=(
                request.thread_id
            ),
            role="assistant",
            content=result["answer"],
            sources=sources,
            route=result.get(
                "route"
            ),
            retrieval_relevant=(
                result.get(
                    "retrieval_relevant"
                )
            ),
        )

        logger.info(
            (
                "Chat request completed: "
                "thread_id=%s "
                "route=%s relevant=%s"
            ),
            request.thread_id,
            result.get(
                "route"
            ),
            result.get(
                "retrieval_relevant"
            ),
        )

        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            thread_id=(
                request.thread_id
            ),
            route=result["route"],
            retrieval_relevant=(
                result.get(
                    "retrieval_relevant"
                )
            ),
            standalone_question=(
                result.get(
                    "standalone_question"
                )
            ),
            sources=sources,
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            (
                "Chat endpoint failed: "
                "thread_id=%s"
            ),
            request.thread_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG assistant "
                "failed to process "
                "the request."
            ),
        )