from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.retrieval.retriever import create_vector_store


# --------------------------------------------------
# Basic tokenizer
# --------------------------------------------------

def tokenize(text):
    """
    Convert text into lowercase tokens for BM25.
    """
    return text.lower().split()


# --------------------------------------------------
# Document identity
# --------------------------------------------------

def document_key(document):
    """
    Create a unique key for each chunk.

    This helps us merge the same chunk returned
    by vector search and BM25 search.
    """

    metadata = document.metadata

    return (
        metadata.get("source"),
        metadata.get("chunk_id"),
        metadata.get("page_label"),
    )


# --------------------------------------------------
# Get all corpus documents from Chroma
# --------------------------------------------------

def get_corpus_documents(
    metadata_filter=None,
):
    """
    Load indexed chunks directly from Chroma.

    If metadata_filter is supplied,
    BM25 only searches inside that scope.
    """

    vector_store = create_vector_store()

    if metadata_filter:

        data = vector_store.get(
            where=metadata_filter,
            include=[
                "documents",
                "metadatas",
            ],
        )

    else:

        data = vector_store.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

    documents = []

    for text, metadata in zip(
        data["documents"],
        data["metadatas"],
    ):

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


# --------------------------------------------------
# BM25 keyword search
# --------------------------------------------------

def bm25_search(
    query,
    metadata_filter=None,
    k=5,
):
    """
    Retrieve documents using BM25 keyword search.
    """

    documents = get_corpus_documents(
        metadata_filter=metadata_filter
    )

    if not documents:
        return []

    tokenized_corpus = [
        tokenize(document.page_content)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    tokenized_query = tokenize(
        query
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        document
        for document, score
        in ranked_results[:k]
    ]


# --------------------------------------------------
# Vector semantic search
# --------------------------------------------------

def vector_search(
    query,
    metadata_filter=None,
    k=5,
):
    """
    Retrieve documents using semantic vector search.
    """

    vector_store = create_vector_store()

    if metadata_filter:

        documents = vector_store.similarity_search(
            query,
            k=k,
            filter=metadata_filter,
        )

    else:

        documents = vector_store.similarity_search(
            query,
            k=k,
        )

    return documents


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    vector_documents,
    bm25_documents,
    rrf_k=60,
):
    """
    Combine vector search and BM25 rankings using
    Reciprocal Rank Fusion.

    Formula:

        RRF Score = 1 / (rrf_k + rank)

    A document appearing in both search results
    receives score contributions from both.
    """

    scores = {}
    document_map = {}


    # Vector results
    for rank, document in enumerate(
        vector_documents,
        start=1,
    ):

        key = document_key(
            document
        )

        document_map[key] = document

        scores[key] = (
            scores.get(key, 0)
            + 1 / (rrf_k + rank)
        )


    # BM25 results
    for rank, document in enumerate(
        bm25_documents,
        start=1,
    ):

        key = document_key(
            document
        )

        document_map[key] = document

        scores[key] = (
            scores.get(key, 0)
            + 1 / (rrf_k + rank)
        )


    ranked_keys = sorted(
        scores.keys(),
        key=lambda key: scores[key],
        reverse=True,
    )

    fused_documents = [
        document_map[key]
        for key in ranked_keys
    ]

    return fused_documents


# --------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------

def hybrid_retrieve_documents(
    query,
    metadata_filter=None,
    vector_k=5,
    bm25_k=5,
    final_k=5,
):
    """
    Hybrid retrieval pipeline:

        Query
          ↓
    ┌───────────────┐
    │               │
    ↓               ↓
    Vector          BM25
    Search          Search
    │               │
    └───────┬───────┘
            ↓
    Reciprocal Rank Fusion
            ↓
       Top Candidates
    """

    vector_documents = vector_search(
        query,
        metadata_filter=metadata_filter,
        k=vector_k,
    )

    bm25_documents = bm25_search(
        query,
        metadata_filter=metadata_filter,
        k=bm25_k,
    )

    fused_documents = reciprocal_rank_fusion(
        vector_documents,
        bm25_documents,
    )

    return fused_documents[:final_k]


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    test_queries = [
        "What does NovaSearch do?",
        "What is the refund policy?",
        "How are customer passwords protected?",
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        documents = hybrid_retrieve_documents(
            query,
            final_k=5,
        )

        for index, document in enumerate(
            documents,
            start=1,
        ):

            metadata = document.metadata

            print(
                f"\nRank {index}"
            )

            print(
                "Source:",
                metadata.get("source"),
            )

            print(
                "Chunk:",
                metadata.get("chunk_id"),
            )

            if (
                metadata.get("page_label")
                is not None
            ):

                print(
                    "Page:",
                    metadata.get(
                        "page_label"
                    ),
                )

            print(
                "Text:",
                document.page_content[:200],
            )