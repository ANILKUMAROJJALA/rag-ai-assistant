from typing import TypedDict

from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from app.retrieval.retriever import create_retriever
from app.retrieval.reranker import create_reranker, rerank_documents
from app.generation.llm import create_llm
from app.generation.prompts import RAG_PROMPT


# ---------------------------------------------------------
# Initialize expensive components once
# ---------------------------------------------------------

retriever = create_retriever()
reranker = create_reranker()
llm = create_llm()

connection = sqlite3.connect(
    "rag_checkpoints.sqlite",
    check_same_thread=False,
)

checkpointer = SqliteSaver(connection)


# ---------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------

class RAGState(TypedDict):
    question: str
    standalone_question: str
    conversation_history: list
    retrieved_documents: list
    reranked_documents: list
    answer: str
    sources: list
    route: str


# ---------------------------------------------------------
# Route Node
# ---------------------------------------------------------

def route_question_node(state: RAGState):
    """Decide whether the question needs document retrieval."""

    routing_prompt = f"""
Classify the user's message into exactly one category:

rag
direct

Use "rag" when the user is asking for information that should come
from the company documents.

Use "direct" for greetings, thanks, casual conversation,
or general conversational messages that do not require the documents.

User message:
{state["question"]}

Return only:
rag
or
direct
"""

    response = llm.invoke(routing_prompt)

    route = response.content.strip().lower()

    if route not in {"rag", "direct"}:
        route = "rag"

    return {
        "route": route
    }


def choose_route(state: RAGState):
    """Return the route selected by the routing node."""

    return state["route"]


# ---------------------------------------------------------
# Direct Response Node
# ---------------------------------------------------------

def direct_response_node(state: RAGState):
    """Answer conversational questions without document retrieval."""

    prompt = f"""
You are a helpful AI assistant.

Respond naturally and briefly to the user's conversational message.

User:
{state["question"]}
"""

    response = llm.invoke(prompt)

    updated_history = state["conversation_history"].copy()

    updated_history.append(
        f"User: {state['question']}"
    )

    updated_history.append(
        f"Assistant: {response.content}"
    )

    return {
        "answer": response.content,
        "sources": [],
        "conversation_history": updated_history,
    }


# ---------------------------------------------------------
# Rewrite Query Node
# ---------------------------------------------------------

def rewrite_query_node(state: RAGState):
    """Rewrite a follow-up question into a standalone question."""

    history = state["conversation_history"]

    if not history:
        return {
            "standalone_question": state["question"]
        }

    history_text = "\n".join(history)

    rewrite_prompt = f"""
Rewrite the user's latest question as a standalone question
using the conversation history.

Conversation history:
{history_text}

Latest question:
{state["question"]}

Return only the rewritten standalone question.
"""

    response = llm.invoke(rewrite_prompt)

    return {
        "standalone_question": response.content.strip()
    }


# ---------------------------------------------------------
# Retrieve Node
# ---------------------------------------------------------

def retrieve_node(state: RAGState):
    """Retrieve relevant documents."""

    documents = retriever.invoke(
        state["standalone_question"]
    )

    return {
        "retrieved_documents": documents
    }


# ---------------------------------------------------------
# Rerank Node
# ---------------------------------------------------------

def rerank_node(state: RAGState):
    """Rerank retrieved documents based on relevance."""

    ranked_documents = rerank_documents(
        state["standalone_question"],
        state["retrieved_documents"],
        reranker,
    )

    reranked_documents = [
        document
        for document, score in ranked_documents
    ]

    return {
        "reranked_documents": reranked_documents
    }


# ---------------------------------------------------------
# Context Formatting
# ---------------------------------------------------------

def format_context(documents):
    """Combine reranked documents into context for the LLM."""

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ---------------------------------------------------------
# Generate Node
# ---------------------------------------------------------

def generate_node(state: RAGState):
    """Generate a grounded RAG answer."""

    top_documents = state["reranked_documents"][:2]

    context = format_context(top_documents)

    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": state["question"],
        }
    )

    response = llm.invoke(prompt)

    sources = []

    for document in top_documents:
        metadata = document.metadata

        source_info = {
            "source": metadata.get("source"),
            "file_type": metadata.get("file_type"),
            "chunk_id": metadata.get("chunk_id"),
        }

        if metadata.get("page_label") is not None:
            source_info["page"] = metadata.get("page_label")

        sources.append(source_info)

    updated_history = state["conversation_history"].copy()

    updated_history.append(
        f"User: {state['question']}"
    )

    updated_history.append(
        f"Assistant: {response.content}"
    )

    return {
        "answer": response.content,
        "sources": sources,
        "conversation_history": updated_history,
    }


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_graph():
    """Build and compile the conversational RAG workflow."""

    graph = StateGraph(RAGState)

    graph.add_node(
        "route_question",
        route_question_node,
    )

    graph.add_node(
        "direct_response",
        direct_response_node,
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query_node,
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "rerank",
        rerank_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_edge(
        START,
        "route_question",
    )

    graph.add_conditional_edges(
        "route_question",
        choose_route,
        {
            "direct": "direct_response",
            "rag": "rewrite_query",
        },
    )

    graph.add_edge(
        "direct_response",
        END,
    )

    graph.add_edge(
        "rewrite_query",
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "rerank",
    )

    graph.add_edge(
        "rerank",
        "generate",
    )

    graph.add_edge(
        "generate",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer
    )


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    rag_graph = build_graph()

    config = {
        "configurable": {
            "thread_id": "sqlite-proof-final"
        }
    }

    follow_up_input = {
    "question": "What is TechNova AI's refund policy?",
    "standalone_question": "",
}

    

    result = rag_graph.invoke(
    follow_up_input,
    config=config,
)


    print("\nQuestion:")
    print(result["question"])

    print("\nRoute:")
    print(result["route"])

    print("\nStandalone Question:")
    print(result["standalone_question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(result["sources"])

    print("\nConversation History:")
    for message in result["conversation_history"]:
        print(message)