from app.ingestion.embeddings import create_embedding_model
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


if __name__ == "__main__":
    vector_store = create_vector_store()

    correct_at_1 = 0

    for test_case in TEST_CASES:
        query = test_case["query"]
        expected_chunk = test_case["expected_chunk"]

        results = vector_store.similarity_search_with_score(
            query,
            k=3,
        )

        retrieved_chunks = [
            document.metadata.get("chunk_id")
            for document, score in results
        ]

        rank = (
            retrieved_chunks.index(expected_chunk) + 1
            if expected_chunk in retrieved_chunks
            else None
        )

        if rank == 1:
            correct_at_1 += 1

        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print(f"Expected chunk: {expected_chunk}")
        print(f"Retrieved ranking: {retrieved_chunks}")
        print(f"Expected chunk rank: {rank}")

    total = len(TEST_CASES)
    recall_at_1 = correct_at_1 / total

    print("\n" + "=" * 70)
    print("RETRIEVAL BASELINE")
    print("=" * 70)
    print(f"Correct at rank 1: {correct_at_1}/{total}")
    print(f"Recall@1: {recall_at_1:.2%}")