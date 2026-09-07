from typing import TypedDict

from langgraph.graph import StateGraph, START, END

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


# ---------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------

class RAGState(TypedDict):
    question: str
    conversation_history: list
    retrieved_documents: list
    reranked_documents: list
    answer: str
    sources: list


# ---------------------------------------------------------
# Retrieve Node
# ---------------------------------------------------------

def retrieve_node(state: RAGState):
    """Retrieve relevant documents for the user's question."""

    documents = retriever.invoke(
        state["question"]
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
        state["question"],
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
    """Generate a grounded answer using the reranked documents."""

    context = format_context(
        state["reranked_documents"]
    )

    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": state["question"],
        }
    )

    response = llm.invoke(prompt)

    sources = [
        {
            "source": document.metadata.get("source"),
            "chunk_id": document.metadata.get("chunk_id"),
        }
        for document in state["reranked_documents"]
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_graph():
    """Build and compile the RAG workflow."""

    graph = StateGraph(RAGState)

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

    return graph.compile()


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    rag_graph = build_graph()

    initial_state = {
        "question": "Where is TechNova AI headquartered?",
        "conversation_history": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "answer": "",
        "sources": [],
    }

    result = rag_graph.invoke(
        initial_state
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(chunk {source['chunk_id']})"
        )