import hashlib

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


@lru_cache(maxsize=1)
def get_live_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_live_vector_store():
    return Chroma(
        collection_name=(
            CHROMA_COLLECTION_NAME
        ),
        persist_directory=(
            CHROMA_PERSIST_DIRECTORY
        ),
        embedding_function=(
            get_live_embeddings()
        ),
    )


def load_uploaded_document(
    file_path: Path,
):
    extension = (
        file_path.suffix.lower()
    )

    if extension == ".txt":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    elif extension == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

    elif extension == ".docx":

        loader = Docx2txtLoader(
            str(file_path)
        )

    else:
        raise ValueError(
            (
                "Unsupported file type. "
                "Only TXT, PDF, and DOCX "
                "are supported."
            )
        )

    documents = loader.load()

    source_name = (
        file_path.name
    )

    file_type = (
        extension.replace(
            ".",
            "",
        )
    )

    for document in documents:

        document.metadata[
            "source"
        ] = source_name

        document.metadata[
            "file_type"
        ] = file_type

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


def create_stable_chunk_id(
    document,
):
    source = (
        document.metadata.get(
            "source",
            "",
        )
    )

    page = (
        document.metadata.get(
            "page_label",
            document.metadata.get(
                "page",
                "",
            ),
        )
    )

    raw_value = (
        f"{source}|"
        f"{page}|"
        f"{document.page_content}"
    )

    return hashlib.sha256(
        raw_value.encode(
            "utf-8"
        )
    ).hexdigest()


def chunk_uploaded_documents(
    documents,
):
    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=(
                CHUNK_OVERLAP
            ),
        )
    )

    chunks = (
        splitter.split_documents(
            documents
        )
    )

    for chunk in chunks:

        chunk_id = (
            create_stable_chunk_id(
                chunk
            )
        )

        chunk.metadata[
            "chunk_id"
        ] = chunk_id

    return chunks


def delete_source_from_chroma(
    source_name: str,
):
    vector_store = (
        get_live_vector_store()
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

    if ids:
        vector_store.delete(
            ids=ids
        )

    return len(ids)


def index_file(
    file_path: Path,
):
    if (
        file_path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):
        raise ValueError(
            "Unsupported file type."
        )

    documents = (
        load_uploaded_document(
            file_path
        )
    )

    chunks = (
        chunk_uploaded_documents(
            documents
        )
    )

    if not chunks:
        raise ValueError(
            (
                "The uploaded document "
                "did not contain readable "
                "text."
            )
        )

    source_name = (
        file_path.name
    )

    delete_source_from_chroma(
        source_name
    )

    vector_store = (
        get_live_vector_store()
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
        "source":
            source_name,

        "chunks":
            len(chunks),
    }