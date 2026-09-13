from pathlib import Path

from app.ingestion.chunking import (
    split_documents,
)

from app.ingestion.loader import (
    SUPPORTED_EXTENSIONS,
    load_document,
)

from app.ingestion.vectorstore import (
    create_vector_store,
)


def delete_source_from_chroma(
    source_name: str,
):
    """
    Delete all vector chunks belonging
    to one source document.

    Returns the number of deleted chunks.
    """

    vector_store = (
        create_vector_store()
    )

    result = vector_store.get(
        where={
            "source": source_name
        }
    )

    ids = result.get(
        "ids",
        [],
    )

    if not ids:
        return 0

    vector_store.delete(
        ids=ids
    )

    return len(ids)


def index_file(
    file_path: Path,
):
    """
    Load, chunk, and index one document.

    This is the production ingestion path
    used by the document-upload API.
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = (
        file_path.suffix.lower()
    )

    if (
        extension
        not in SUPPORTED_EXTENSIONS
    ):
        raise ValueError(
            "Unsupported file type. "
            "Only TXT, PDF, and DOCX "
            "are supported."
        )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    documents = load_document(
        str(file_path)
    )

    # -----------------------------------------------------
    # Chunk + deterministic IDs
    # -----------------------------------------------------

    chunks = split_documents(
        documents
    )

    if not chunks:
        raise ValueError(
            "The uploaded document "
            "did not contain readable text."
        )

    source_name = (
        file_path.name
    )

    # -----------------------------------------------------
    # Remove stale vectors for this source
    # -----------------------------------------------------

    delete_source_from_chroma(
        source_name
    )

    # -----------------------------------------------------
    # Store new chunks
    # -----------------------------------------------------

    vector_store = (
        create_vector_store()
    )

    ids = [
        chunk.metadata[
            "chunk_id"
        ]
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return {
        "source": source_name,
        "chunks": len(chunks),
    }