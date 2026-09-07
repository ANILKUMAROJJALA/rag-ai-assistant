from app.ingestion.embeddings import create_embedding_model
from app.retrieval.reranker import create_reranker, rerank_documents
from langchain_chroma import Chroma


def create_vector_store():
    embeddings = create_embedding_model()

    return Chroma(
        collection_name="technova_documents",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    vector_store = create_vector_store()
    reranker = create_reranker()

    query = "What technologies does TechNova AI use?"

    retrieved = vector_store.similarity_search(
        query,
        k=3,
    )

    results = rerank_documents(
        query,
        retrieved,
        reranker,
    )

    print(f"Query: {query}\n")

    for rank, (document, score) in enumerate(results, start=1):
        print(f"--- Reranked Result {rank} ---")
        print(f"Reranker score: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Chunk ID: {document.metadata.get('chunk_id')}")
        print("\nContent:")
        print(document.page_content)
        print()