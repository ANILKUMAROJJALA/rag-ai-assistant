from app.generation.llm import create_llm
from app.generation.prompts import RAG_PROMPT
from app.retrieval.reranker import create_reranker, rerank_documents
from app.retrieval.retriever import create_retriever


def create_rag_pipeline():
    """Create the components required for the RAG pipeline."""

    retriever = create_retriever()
    reranker = create_reranker()
    llm = create_llm()

    return retriever, reranker, llm


def format_context(documents):
    """Combine retrieved documents into a single context string."""

    return "\n\n".join(
        document.page_content
        for document in documents
    )


def answer_question(question: str):
    """Retrieve, rerank, generate an answer, and return sources."""

    retriever, reranker, llm = create_rag_pipeline()

    # Step 1: Retrieve candidate documents
    documents = retriever.invoke(question)

    # Step 2: Rerank retrieved documents
    ranked_documents = rerank_documents(
        question,
        documents,
        reranker,
    )

    # Step 3: Keep reranked documents
    reranked_docs = [
        document
        for document, score in ranked_documents
    ]

    # Step 4: Build context
    context = format_context(reranked_docs)

    # Step 5: Build the prompt
    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # Step 6: Generate answer
    response = llm.invoke(prompt)

    # Step 7: Extract source metadata
    sources = [
        {
            "source": document.metadata.get("source"),
            "chunk_id": document.metadata.get("chunk_id"),
        }
        for document in reranked_docs
    ]

    return response.content, sources


if __name__ == "__main__":
    question = "Where is TechNova AI headquartered?"

    answer, sources = answer_question(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for source in sources:
        print(
            f"- {source['source']} "
            f"(chunk {source['chunk_id']})"
        )