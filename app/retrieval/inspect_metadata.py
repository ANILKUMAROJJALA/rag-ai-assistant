from langchain_chroma import Chroma

from app.ingestion.embeddings import create_embedding_model


def inspect_metadata():
    embeddings = create_embedding_model()

    vector_store = Chroma(
        collection_name="technova_documents",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )

    data = vector_store.get(
        include=["metadatas", "documents"]
    )

    print(f"\nTotal chunks: {len(data['ids'])}")

    for index, (chunk_id, metadata, document) in enumerate(
        zip(
            data["ids"],
            data["metadatas"],
            data["documents"],
        ),
        start=1,
    ):
        print("\n" + "=" * 70)
        print(f"CHUNK {index}")
        print("=" * 70)

        print("Chroma ID:", chunk_id)
        print("Metadata:", metadata)
        print("Text:", document[:200])


if __name__ == "__main__":
    inspect_metadata()