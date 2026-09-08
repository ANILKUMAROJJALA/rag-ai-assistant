from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """Split documents into chunks while preserving source metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

        # Preserve file_type, page, source, etc.
        source = chunk.metadata.get("source", "")
        chunk.metadata["source"] = source.split("\\")[-1]

    return chunks


if __name__ == "__main__":
    from app.ingestion.loader import load_documents_from_folder

    documents = load_documents_from_folder(
        "data/raw"
    )

    chunks = split_documents(documents)

    print(f"\nOriginal document units: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n--- Chunk ---")
        print("Metadata:", chunk.metadata)
        print("Content:")
        print(chunk.page_content[:300])