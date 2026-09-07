from app.ingestion.embeddings import create_embedding_model
from app.retrieval.reranker import create_reranker, rerank_documents
from langchain_chroma import Chroma


TEST_CASES = [
    {
        "query": "Where is TechNova AI headquartered?",
        "expected_chunk": 0,
    },
    {
        "query": "What products does TechNova AI offer?",
        "expected_chunk": 0,
    },
    {
        "query": "What are the customer support hours?",
        "expected_chunk": 1,
    },
    {
        "query": "What is the refund policy?",
        "expected_chunk": 1,
    },
    {
        "query": "What technologies does TechNova AI use?",
        "expected_chunk": 2,
    },
]


def create_vector_store():
    embeddings = create_embedding_model()

    return Chroma(
        collection_name="technova_documents",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )


def get_rank(documents, expected_chunk):
    chunk_ids = [
        document.metadata.get("chunk_id")
        for document in documents
    ]

    if expected_chunk in chunk_ids:
        return chunk_ids.index(expected_chunk) + 1

    return None


if __name__ == "__main__":
    vector_store = create_vector_store()
    reranker = create_reranker()

    vector_correct_at_1 = 0
    reranked_correct_at_1 = 0

    for test_case in TEST_CASES:
        query = test_case["query"]
        expected_chunk = test_case["expected_chunk"]

        retrieved = vector_store.similarity_search(
            query,
            k=3,
        )

        vector_rank = get_rank(
            retrieved,
            expected_chunk,
        )

        reranked = rerank_documents(
            query,
            retrieved,
            reranker,
        )

        reranked_documents = [
            document
            for document, score in reranked
        ]

        reranked_rank = get_rank(
            reranked_documents,
            expected_chunk,
        )

        if vector_rank == 1:
            vector_correct_at_1 += 1

        if reranked_rank == 1:
            reranked_correct_at_1 += 1

        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print(f"Expected chunk: {expected_chunk}")
        print(f"Vector rank: {vector_rank}")
        print(f"Reranked rank: {reranked_rank}")

    total = len(TEST_CASES)

    print("\n" + "=" * 70)
    print("RERANKING EVALUATION")
    print("=" * 70)

    print(
        f"Vector Recall@1: "
        f"{vector_correct_at_1}/{total} "
        f"({vector_correct_at_1 / total:.2%})"
    )

    print(
        f"Reranked Recall@1: "
        f"{reranked_correct_at_1}/{total} "
        f"({reranked_correct_at_1 / total:.2%})"
    )