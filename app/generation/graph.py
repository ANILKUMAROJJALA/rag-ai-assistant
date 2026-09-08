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
    standalone_question: str
    conversation_history: list
    retrieved_documents: list
    reranked_documents: list
    answer: str
    sources: list

# ---------------------------------------------------------
# Retrieve Node
# ---------------------------------------------------------

def retrieve_node(state: RAGState):
    """Retrieve relevant documents."""

    query = state["standalone_question"]

    documents = retriever.invoke(query)

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
    """Generate a grounded answer from the highest-ranked document."""

    top_documents = state["reranked_documents"][:1]

    context = format_context(top_documents)

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
        for document in top_documents
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }

def rewrite_query_node(state: RAGState):
    """Rewrite a follow-up question into a standalone question."""

    history = state["conversation_history"]

    if not history:
        return {
            "standalone_question": state["question"]
        }

    history_text = "\n".join(history)

    rewrite_prompt = f"""
Rewrite the user's latest question as a standalone question using the conversation history.

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
# Build LangGraph
# ---------------------------------------------------------

def build_graph():
    """Build and compile the conversational RAG workflow."""

    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    rag_graph = build_graph()

    conversation_history = []

    # -------------------------
    # First question
    # -------------------------

    first_state = {
        "question": "Where is TechNova AI headquartered?",
        "standalone_question": "",
        "conversation_history": conversation_history,
        "retrieved_documents": [],
        "reranked_documents": [],
        "answer": "",
        "sources": [],
    }

    first_result = rag_graph.invoke(first_state)

    print("\nQuestion 1:")
    print(first_result["question"])

    print("\nStandalone Question 1:")
    print(first_result["standalone_question"])

    print("\nAnswer 1:")
    print(first_result["answer"])

    print("\nSources 1:")
    for source in first_result["sources"]:
        print(
            f"- {source['source']} "
            f"(chunk {source['chunk_id']})"
        )

    # Save first turn into conversation history
    conversation_history.append(
        f"User: {first_result['question']}"
    )

    conversation_history.append(
        f"Assistant: {first_result['answer']}"
    )

    # -------------------------
    # Follow-up question
    # -------------------------

    second_state = {
        "question": "What technologies do they use?",
        "standalone_question": "",
        "conversation_history": conversation_history,
        "retrieved_documents": [],
        "reranked_documents": [],
        "answer": "",
        "sources": [],
    }

    second_result = rag_graph.invoke(second_state)

    print("\n" + "=" * 60)

    print("\nQuestion 2:")
    print(second_result["question"])

    print("\nStandalone Question 2:")
    print(second_result["standalone_question"])

    print("\nAnswer 2:")
    print(second_result["answer"])

    print("\nSources 2:")
    for source in second_result["sources"]:
        print(
            f"- {source['source']} "
            f"(chunk {source['chunk_id']})"
        )