from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """Split documents into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    from loader import load_text_file

    documents = load_text_file("data/raw/company_info.txt")
    chunks = split_documents(documents)

    print(f"Original documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content)