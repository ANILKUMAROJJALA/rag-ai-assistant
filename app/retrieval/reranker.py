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