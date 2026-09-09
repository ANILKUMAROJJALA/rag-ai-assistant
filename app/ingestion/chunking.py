import hashlib

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


def create_stable_chunk_id(
    source,
    page,
    text,
):
    """
    Create a deterministic chunk ID.

    The same source + page + chunk text
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
    Split documents into chunks and assign
    deterministic chunk IDs.
    """

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
    )

    chunks = splitter.split_documents(
        documents
    )

    for chunk in chunks:

        source = chunk.metadata.get(
            "source",
            "unknown"
        )

        # Normalize Windows paths
        source = (
            source
            .replace("\\", "/")
            .split("/")[-1]
        )

        chunk.metadata[
            "source"
        ] = source


        # PDF page information
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


        # Store stable ID in metadata
        chunk.metadata[
            "chunk_id"
        ] = stable_id


    return chunks