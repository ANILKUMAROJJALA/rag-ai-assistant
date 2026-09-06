from pathlib import Path

from langchain_community.document_loaders import TextLoader


def load_text_file(file_path: str):
    """Load a text file and return LangChain documents."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(str(path), encoding="utf-8")

    documents = loader.load()

    return documents


if __name__ == "__main__":
    file_path = "data/raw/company_info.txt"

    documents = load_text_file(file_path)

    print(f"Loaded {len(documents)} document(s).")
    print("\nFirst document:")
    print(documents[0].page_content)