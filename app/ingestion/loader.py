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

    Metadata is normalized so the rest of the
    application receives a consistent source name,
    file type, and human-readable page label.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    if extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

    elif extension == ".pdf":
        loader = PyPDFLoader(
            str(path)
        )

    else:
        loader = Docx2txtLoader(
            str(path)
        )

    documents = loader.load()

    # -----------------------------------------------------
    # Normalize metadata
    # -----------------------------------------------------

    for document in documents:

        document.metadata[
            "source"
        ] = path.name

        document.metadata[
            "file_type"
        ] = extension.replace(
            ".",
            "",
        )

        # PyPDFLoader normally uses a zero-based
        # "page" value. page_label is what we want
        # to display to the user.
        if (
            document.metadata.get(
                "page_label"
            )
            is None
            and document.metadata.get(
                "page"
            )
            is not None
        ):
            document.metadata[
                "page_label"
            ] = str(
                int(
                    document.metadata[
                        "page"
                    ]
                )
                + 1
            )

    return documents


# ---------------------------------------------------------
# Load All Documents From Folder
# ---------------------------------------------------------

def load_documents_from_folder(
    folder_path: str,
):
    """
    Load all supported documents from a folder.

    This helper remains available for development,
    demos, and backward compatibility.
    """

    folder = Path(
        folder_path
    )

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
                "Skipping unsupported file: "
                f"{file_path.name}"
            )
            continue

        print(
            f"Loading: {file_path.name}"
        )

        documents = load_document(
            str(file_path)
        )

        all_documents.extend(
            documents
        )

    return all_documents


# ---------------------------------------------------------
# Manual Development Test
# ---------------------------------------------------------

if __name__ == "__main__":

    documents = (
        load_documents_from_folder(
            "data/raw"
        )
    )

    print(
        "\nTotal loaded document units: "
        f"{len(documents)}"
    )

    for i, document in enumerate(
        documents
    ):

        print(
            f"\n--- Document {i + 1} ---"
        )

        print(
            "Source:",
            document.metadata.get(
                "source"
            ),
        )

        print(
            "File type:",
            document.metadata.get(
                "file_type"
            ),
        )

        print(
            "Page:",
            document.metadata.get(
                "page_label",
                document.metadata.get(
                    "page"
                ),
            ),
        )

        print(
            "Preview:",
            document.page_content[:200],
        )