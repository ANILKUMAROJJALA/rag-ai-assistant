from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def create_reranker():
    """Create the local cross-encoder reranker."""

    return CrossEncoder(MODEL_NAME)


def rerank_documents(query, documents, reranker):
    """Rerank documents according to query-document relevance."""

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_documents


if __name__ == "__main__":
    reranker = create_reranker()

    query = "What technologies does TechNova AI use?"

    documents = [
        "TechNova AI is a fictional software company founded in 2021.",
        "TechNova AI uses Python, FastAPI, PostgreSQL, Docker, Kubernetes, and cloud infrastructure.",
        "Customer support is available Monday through Friday.",
    ]

    results = rerank_documents(query, documents, reranker)

    for document, score in results:
        print(f"Score: {score:.4f}")
        print(document)
        print()