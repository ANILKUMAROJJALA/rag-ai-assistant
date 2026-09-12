from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL


def create_reranker():
    """
    Create the cross-encoder reranker.
    """

    return CrossEncoder(
        RERANKER_MODEL
    )


def rerank_documents(
    query,
    documents,
    reranker,
):
    """
    Rerank retrieved documents using
    query-document cross-encoder scores.
    """

    if not documents:
        return []

    pairs = [
        (
            query,
            document.page_content,
        )
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked_documents = sorted(
        zip(
            documents,
            scores,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_documents