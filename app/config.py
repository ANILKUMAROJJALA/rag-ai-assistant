import os

from dotenv import load_dotenv


load_dotenv()


# ======================================================
# LLM CONFIGURATION
# ======================================================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-5.6",
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0",
    )
)


# ======================================================
# EMBEDDING CONFIGURATION
# ======================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ======================================================
# RERANKER CONFIGURATION
# ======================================================

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# ======================================================
# CHROMA CONFIGURATION
# ======================================================

CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    "chroma_db",
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "technova_documents",
)


# ======================================================
# INGESTION CONFIGURATION
# ======================================================

RAW_DATA_DIRECTORY = os.getenv(
    "RAW_DATA_DIRECTORY",
    "data/raw",
)

INGESTION_MANIFEST_PATH = os.getenv(
    "INGESTION_MANIFEST_PATH",
    "data/processed/ingestion_manifest.json",
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "500",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "50",
    )
)


# ======================================================
# RETRIEVAL CONFIGURATION
# ======================================================

VECTOR_TOP_K = int(
    os.getenv(
        "VECTOR_TOP_K",
        "5",
    )
)

BM25_TOP_K = int(
    os.getenv(
        "BM25_TOP_K",
        "5",
    )
)

HYBRID_FINAL_K = int(
    os.getenv(
        "HYBRID_FINAL_K",
        "8",
    )
)

GENERATION_TOP_K = int(
    os.getenv(
        "GENERATION_TOP_K",
        "2",
    )
)

RRF_K = int(
    os.getenv(
        "RRF_K",
        "60",
    )
)


# ======================================================
# RELEVANCE CONFIGURATION
# ======================================================

RELEVANCE_THRESHOLD = float(
    os.getenv(
        "RELEVANCE_THRESHOLD",
        "0.0",
    )
)


# ======================================================
# LANGGRAPH / CHECKPOINT CONFIGURATION
# ======================================================

CHECKPOINT_DATABASE = os.getenv(
    "CHECKPOINT_DATABASE",
    "rag_checkpoints.sqlite",
)