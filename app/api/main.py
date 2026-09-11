from functools import lru_cache

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="RAG AI Assistant API",
    description=(
        "API for the production-style "
        "RAG AI Assistant."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# --------------------------------------------------
# RAG application dependency
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_rag_app():
    """
    Lazily import and build the LangGraph
    RAG application.

    This prevents expensive RAG modules from
    loading when endpoints such as /health
    are being used or tested.

    The graph is created only on the first
    request that actually needs it, then the
    same instance is reused.
    """

    from app.generation.graph import build_graph

    return build_graph()


# --------------------------------------------------
# Initial state
# --------------------------------------------------

def create_initial_state(
    question: str,
):
    """
    Create the complete LangGraph state
    for a brand-new conversation.
    """

    return {
        "question":
            question,

        "standalone_question":
            "",

        "conversation_history":
            [],

        "retrieved_documents":
            [],

        "reranked_documents":
            [],

        "reranker_scores":
            [],

        "answer":
            "",

        "sources":
            [],

        "route":
            "",

        "metadata_filter":
            {},

        "retrieval_relevant":
            None,
    }


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

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
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status":
            "healthy",

        "service":
            "RAG AI Assistant",
    }


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    rag_app=Depends(get_rag_app),
):
    """
    Send a user question to the
    LangGraph RAG pipeline.
    """

    try:

        config = {
            "configurable": {
                "thread_id":
                    request.thread_id
            }
        }

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

        result = (
            rag_app.invoke(
                graph_input,
                config=config,
            )
        )

        return ChatResponse(
            question=
                request.question,

            answer=
                result[
                    "answer"
                ],

            thread_id=
                request.thread_id,

            route=
                result[
                    "route"
                ],

            retrieval_relevant=
                result.get(
                    "retrieval_relevant"
                ),

            standalone_question=
                result.get(
                    "standalone_question"
                ),

            sources=
                result.get(
                    "sources",
                    [],
                ),
        )

    except Exception as error:

        print(
            "Chat endpoint error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG assistant failed "
                "to process the request."
            ),
        ) from error