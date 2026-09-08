from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)


# ---------------------------------------------------------
# Supported File Types
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


# ---------------------------------------------------------
# Load Single Document
# ---------------------------------------------------------

def load_document(file_path: str):
    """
    Load a single document based on its file extension.

    Supported:
    - .txt
    - .pdf
    - .docx
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

    elif extension == ".pdf":
        loader = PyPDFLoader(
            str(path)
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(
            str(path)
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    documents = loader.load()

    # Add our own consistent metadata
    for document in documents:
        document.metadata["source"] = path.name
        document.metadata["file_type"] = (
            extension.replace(".", "")
        )

    return documents


# ---------------------------------------------------------
# Load All Documents From Folder
# ---------------------------------------------------------

def load_documents_from_folder(folder_path: str):
    """
    Load all supported documents from a folder.
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    all_documents = []

    for file_path in folder.iterdir():

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            print(
                f"Skipping unsupported file: "
                f"{file_path.name}"
            )
            continue

        print(f"Loading: {file_path.name}")

        documents = load_document(
            str(file_path)
        )

        all_documents.extend(documents)

    return all_documents


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    documents = load_documents_from_folder(
        "data/raw"
    )

    print(
        f"\nTotal loaded document units: "
        f"{len(documents)}"
    )

    for i, document in enumerate(documents):

        print(f"\n--- Document {i + 1} ---")

        print("Source:")
        print(
            document.metadata.get("source")
        )

        print("File type:")
        print(
            document.metadata.get("file_type")
        )

        print("Page:")
        print(
            document.metadata.get("page")
        )

        print("Preview:")
        print(
            document.page_content[:200]
        )