from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """Split documents into chunks and add useful metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["file_type"] = "txt"

        source = chunk.metadata.get("source", "")
        chunk.metadata["source"] = source.split("\\")[-1]

    return chunks


if __name__ == "__main__":
    from loader import load_text_file

    documents = load_text_file("data/raw/company_info.txt")
    chunks = split_documents(documents)

    print(f"Original documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n--- Chunk ---")
        print("Metadata:", chunk.metadata)
        print(chunk.page_content)