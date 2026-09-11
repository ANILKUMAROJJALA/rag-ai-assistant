import logging

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
# Logger
# --------------------------------------------------

logger = logging.getLogger("uvicorn.error")


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
# CORS
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
    """

    from app.generation.graph import build_graph

    return build_graph()


# --------------------------------------------------
# Initial LangGraph state
# --------------------------------------------------

def create_initial_state(
    question: str,
):
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


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "RAG AI Assistant API",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "RAG AI Assistant",
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

    # Log the start of the request.
    #
    # We intentionally log the thread ID
    # instead of the user's question because
    # the question may contain private data.

    logger.info(
        "Chat request received: thread_id=%s",
        request.thread_id,
    )


    try:

        # --------------------------------------------------
        # LangGraph configuration
        # --------------------------------------------------
        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }


        # --------------------------------------------------
        # Check existing conversation state
        # --------------------------------------------------

        snapshot = rag_app.get_state(
            config
        )


        existing_state = (
            snapshot.values
            if snapshot
            else {}
        )


        # --------------------------------------------------
        # Prepare graph input
        # --------------------------------------------------

        if not existing_state:

            graph_input = create_initial_state(
                request.question
            )

        else:

            # For an existing thread, LangGraph's
            # checkpoint already contains the
            # conversation state.
            #
            # Therefore we only provide the new
            # question.

            graph_input = {
                "question": request.question
            }


        # --------------------------------------------------
        # Execute LangGraph
        # --------------------------------------------------

        result = rag_app.invoke(
            graph_input,
            config=config,
        )


        # --------------------------------------------------
        # Log successful completion
        # --------------------------------------------------

        logger.info(
            (
                "Chat request completed: "
                "thread_id=%s route=%s relevant=%s"
            ),
            request.thread_id,
            result.get("route"),
            result.get(
                "retrieval_relevant"
            ),
        )


        # --------------------------------------------------
        # API response
        # --------------------------------------------------

        return ChatResponse(
            question=request.question,

            answer=result["answer"],

            thread_id=request.thread_id,

            route=result["route"],

            retrieval_relevant=result.get(
                "retrieval_relevant"
            ),

            standalone_question=result.get(
                "standalone_question"
            ),

            sources=result.get(
                "sources",
                [],
            ),
        )


    # --------------------------------------------------
    # Error handling
    # --------------------------------------------------

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
                "The RAG assistant failed "
                "to process the request."
            ),
        )