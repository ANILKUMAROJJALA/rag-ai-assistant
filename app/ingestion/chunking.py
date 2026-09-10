import hashlib

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def create_stable_chunk_id(
    source,
    page,
    text,
):
    """
    Create a deterministic SHA-256 chunk ID.

    The same source, page, and text
    will always generate the same ID.
    """

    raw_value = (
        f"{source}|{page}|{text}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()


def split_documents(documents):
    """
    Split documents into chunks using
    centralized application configuration.

    Stable deterministic chunk IDs are
    added to every chunk.
    """

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )

    chunks = splitter.split_documents(
        documents
    )

    for chunk in chunks:

        source = chunk.metadata.get(
            "source",
            "unknown",
        )

        source = (
            source
            .replace("\\", "/")
            .split("/")[-1]
        )

        chunk.metadata[
            "source"
        ] = source

        page = chunk.metadata.get(
            "page_label"
        )

        if page is None:
            page = chunk.metadata.get(
                "page",
                0,
            )

        stable_id = (
            create_stable_chunk_id(
                source=source,
                page=page,
                text=chunk.page_content,
            )
        )

        chunk.metadata[
            "chunk_id"
        ] = stable_id

    return chunks