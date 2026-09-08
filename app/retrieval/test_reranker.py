from app.retrieval.retriever import create_retriever
from app.retrieval.reranker import (
    create_reranker,
    rerank_documents,
)


def test_query(query, retriever, reranker):

    print("\n" + "=" * 70)
    print(f"Query: {query}")

    # Step 1: Vector retrieval
    documents = retriever.invoke(query)

    print("\nBEFORE RERANKING:")

    for i, document in enumerate(documents, start=1):
        print(
            f"{i}. "
            f"{document.metadata.get('source')} | "
            f"page={document.metadata.get('page_label')} | "
            f"chunk={document.metadata.get('chunk_id')}"
        )

    # Step 2: Cross-encoder reranking
    ranked_documents = rerank_documents(
        query,
        documents,
        reranker,
    )

    print("\nAFTER RERANKING:")

    for i, (document, score) in enumerate(
        ranked_documents,
        start=1,
    ):
        print(
            f"{i}. "
            f"{document.metadata.get('source')} | "
            f"page={document.metadata.get('page_label')} | "
            f"chunk={document.metadata.get('chunk_id')} | "
            f"score={float(score):.4f}"
        )

        print(
            document.page_content[:200]
        )

        print()


if __name__ == "__main__":

    retriever = create_retriever()
    reranker = create_reranker()

    test_queries = [
        "What does NovaSearch do?",
        "What is TechNova AI's refund policy?",
        "How does TechNova AI protect customer passwords?",
    ]

    for query in test_queries:
        test_query(
            query,
            retriever,
            reranker,
        )