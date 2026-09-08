from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from app.retrieval.retriever import create_retriever
from app.retrieval.reranker import create_reranker, rerank_documents
from app.generation.llm import create_llm
from app.generation.prompts import RAG_PROMPT
from langgraph.checkpoint.memory import InMemorySaver

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
    """Generate a grounded answer and update conversation history."""

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

    # -------------------------
    # Turn 1
    # -------------------------

    state = {
        "question": "Where is TechNova AI headquartered?",
        "standalone_question": "",
        "conversation_history": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "answer": "",
        "sources": [],
    }

    result = rag_graph.invoke(state)

    print("\nQuestion 1:")
    print(result["question"])

    print("\nAnswer 1:")
    print(result["answer"])

    print("\nConversation History:")
    for message in result["conversation_history"]:
        print(message)

    # -------------------------
    # Turn 2
    # -------------------------

    result["question"] = "What technologies do they use?"
    result["standalone_question"] = ""

    result = rag_graph.invoke(result)

    print("\n" + "=" * 60)

    print("\nQuestion 2:")
    print(result["question"])

    print("\nStandalone Question 2:")
    print(result["standalone_question"])

    print("\nAnswer 2:")
    print(result["answer"])

    print("\nSources 2:")
    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(chunk {source['chunk_id']})"
        )

    print("\nConversation History:")
    for message in result["conversation_history"]:
        print(message)