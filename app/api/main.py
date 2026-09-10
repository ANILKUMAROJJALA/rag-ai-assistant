from fastapi import (
    FastAPI,
    HTTPException,
)

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
)

from app.generation.graph import (
    build_graph,
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
# Build LangGraph application
# --------------------------------------------------

rag_app = build_graph()


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
            False,
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


        # --------------------------------------------------
        # Check existing conversation state
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
        # New conversation
        # --------------------------------------------------

        if not existing_state:

            graph_input = (
                create_initial_state(
                    request.question
                )
            )


        # --------------------------------------------------
        # Existing conversation
        # --------------------------------------------------

        else:

            graph_input = {
                "question":
                    request.question
            }


        # --------------------------------------------------
        # Run LangGraph
        # --------------------------------------------------

        result = (
            rag_app.invoke(
                graph_input,
                config=config,
            )
        )


        # --------------------------------------------------
        # Return API response
        # --------------------------------------------------

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