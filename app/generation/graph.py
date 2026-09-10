from typing import TypedDict
import sqlite3

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)

from app.config import (
    RELEVANCE_THRESHOLD,
    GENERATION_TOP_K,
    CHECKPOINT_DATABASE,
)

from app.retrieval.hybrid_retriever import (
    hybrid_retrieve_documents,
)

from app.retrieval.retriever import (
    get_available_sources,
)

from app.retrieval.reranker import (
    create_reranker,
    rerank_documents,
)

from app.generation.llm import (
    create_llm,
)

from app.generation.prompts import (
    RAG_PROMPT,
)


# --------------------------------------------------
# Expensive components
# --------------------------------------------------

reranker = create_reranker()

llm = create_llm()


# --------------------------------------------------
# SQLite checkpointing
# --------------------------------------------------

connection = sqlite3.connect(
    CHECKPOINT_DATABASE,
    check_same_thread=False,
)

checkpointer = SqliteSaver(
    connection
)


# --------------------------------------------------
# LangGraph state
# --------------------------------------------------

class RAGState(TypedDict):

    question: str

    standalone_question: str

    conversation_history: list

    retrieved_documents: list

    reranked_documents: list

    reranker_scores: list

    answer: str

    sources: list

    route: str

    metadata_filter: dict

    retrieval_relevant: bool


# --------------------------------------------------
# Route question
# --------------------------------------------------

def route_question_node(
    state: RAGState,
):

    routing_prompt = f"""
Classify the user's message into exactly one category:

rag

direct

Use "rag" when the user is asking a factual question
that may require information from the company documents.

Use "direct" only for greetings, thanks, casual conversation,
or messages that do not require factual knowledge.

Examples:

"Hi" -> direct
"Thanks" -> direct
"How are you?" -> direct

"What is the refund policy?" -> rag
"What does NovaSearch do?" -> rag
"What is the capital of France?" -> rag

The retrieval relevance guard later decides whether
the company documents contain enough evidence.

User message:

{state["question"]}

Return only:

rag

or

direct
"""

    response = llm.invoke(
        routing_prompt
    )

    route = (
        response.content
        .strip()
        .lower()
    )

    if route not in {
        "rag",
        "direct",
    }:

        route = "rag"

    return {
        "route": route
    }


def choose_route(
    state: RAGState,
):

    return state[
        "route"
    ]


# --------------------------------------------------
# Direct response
# --------------------------------------------------

def direct_response_node(
    state: RAGState,
):

    prompt = f"""
You are a helpful AI assistant.

Respond naturally and briefly to the user's
conversational message.

User:

{state["question"]}
"""

    response = llm.invoke(
        prompt
    )

    updated_history = (
        state[
            "conversation_history"
        ].copy()
    )

    updated_history.append(
        f"User: {state['question']}"
    )

    updated_history.append(
        f"Assistant: {response.content}"
    )

    return {
        "answer":
            response.content,

        "sources":
            [],

        "conversation_history":
            updated_history,
    }


# --------------------------------------------------
# Rewrite conversational query
# --------------------------------------------------

def rewrite_query_node(
    state: RAGState,
):

    history = state[
        "conversation_history"
    ]

    if not history:

        return {
            "standalone_question":
                state["question"]
        }

    history_text = "\n".join(
        history
    )

    rewrite_prompt = f"""
Rewrite the user's latest question as a standalone
question using the conversation history.

Conversation history:

{history_text}

Latest question:

{state["question"]}

Return only the rewritten standalone question.
"""

    response = llm.invoke(
        rewrite_prompt
    )

    return {
        "standalone_question":
            response.content.strip()
    }


# --------------------------------------------------
# Metadata filtering
# --------------------------------------------------

def metadata_filter_node(
    state: RAGState,
):

    question = state[
        "standalone_question"
    ]

    question_lower = (
        question.lower()
    )

    available_sources = (
        get_available_sources()
    )

    if not available_sources:

        return {
            "metadata_filter": {}
        }


    # --------------------------------------------------
    # Exact filename fast path
    # --------------------------------------------------

    for source in (
        available_sources
    ):

        if (
            source.lower()
            in question_lower
        ):

            return {
                "metadata_filter": {
                    "source": source
                }
            }


    # --------------------------------------------------
    # Detect document-scoped intent
    # --------------------------------------------------

    document_scope_phrases = [
        "use the",
        "use this",
        "use only",
        "search only",
        "only in",
        "only from",
        "from the document",
        "from this document",
        "from the file",
        "from this file",
        "in the document",
        "in this document",
        "in the file",
        "in this file",
        "check the document",
        "check this document",
        "check the file",
        "check this file",
        "according to the document",
        "according to the file",
    ]

    document_words = [
        "document",
        "file",
        "pdf",
        "docx",
        "txt",
        "guide",
        "handbook",
        "report",
    ]

    has_scope_phrase = any(
        phrase in question_lower
        for phrase
        in document_scope_phrases
    )

    has_document_word = any(
        word in question_lower
        for word
        in document_words
    )


    # --------------------------------------------------
    # Normal query:
    # don't run metadata-selection LLM
    # --------------------------------------------------

    if (
        not has_scope_phrase
        and
        not has_document_word
    ):

        return {
            "metadata_filter": {}
        }


    # --------------------------------------------------
    # Natural-language document mapping
    # --------------------------------------------------

    sources_text = "\n".join(
        f"- {source}"
        for source
        in available_sources
    )

    filter_prompt = f"""
You are selecting whether the user's question refers
to one specific document from the indexed knowledge base.

Available document sources:

{sources_text}

User question:

{question}

Rules:

1. If the user clearly refers to one specific document,
return the exact source filename from the available list.

2. The user does not need to type the exact filename.

Examples:

"refund document"
may refer to:
refund_policy.pdf

"product guide"
may refer to:
product_guide.docx

"company information file"
may refer to:
company_info.txt

3. If you cannot confidently determine one specific document,
return:

none

4. Never invent a filename.

5. Return only one exact filename from the available list,
or return:

none
"""

    response = llm.invoke(
        filter_prompt
    )

    selected_source = (
        response.content.strip()
    )

    if (
        selected_source
        in available_sources
    ):

        return {
            "metadata_filter": {
                "source":
                    selected_source
            }
        }

    return {
        "metadata_filter": {}
    }


# --------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------

def retrieve_node(
    state: RAGState,
):

    documents = (
        hybrid_retrieve_documents(
            state[
                "standalone_question"
            ],

            metadata_filter=state.get(
                "metadata_filter"
            ),
        )
    )

    return {
        "retrieved_documents":
            documents
    }


# --------------------------------------------------
# Cross-encoder reranking
# --------------------------------------------------

def rerank_node(
    state: RAGState,
):

    ranked_documents = (
        rerank_documents(
            state[
                "standalone_question"
            ],

            state[
                "retrieved_documents"
            ],

            reranker,
        )
    )

    reranked_documents = [
        document
        for document, score
        in ranked_documents
    ]

    reranker_scores = [
        float(score)
        for document, score
        in ranked_documents
    ]

    return {
        "reranked_documents":
            reranked_documents,

        "reranker_scores":
            reranker_scores,
    }


# --------------------------------------------------
# Relevance guard
# --------------------------------------------------

def relevance_guard_node(
    state: RAGState,
):

    scores = state[
        "reranker_scores"
    ]

    if not scores:

        return {
            "retrieval_relevant":
                False
        }

    top_score = scores[0]

    is_relevant = (
        top_score
        >= RELEVANCE_THRESHOLD
    )

    return {
        "retrieval_relevant":
            is_relevant
    }


def choose_relevance(
    state: RAGState,
):

    if state[
        "retrieval_relevant"
    ]:

        return "relevant"

    return "irrelevant"


# --------------------------------------------------
# No-answer response
# --------------------------------------------------

def no_answer_node(
    state: RAGState,
):

    answer = (
        "I don't have enough information "
        "in the provided documents "
        "to answer that."
    )

    updated_history = (
        state[
            "conversation_history"
        ].copy()
    )

    updated_history.append(
        f"User: {state['question']}"
    )

    updated_history.append(
        f"Assistant: {answer}"
    )

    return {
        "answer":
            answer,

        "sources":
            [],

        "conversation_history":
            updated_history,
    }


# --------------------------------------------------
# Context formatting
# --------------------------------------------------

def format_context(
    documents,
):

    return "\n\n".join(
        document.page_content
        for document
        in documents
    )


# --------------------------------------------------
# Grounded generation
# --------------------------------------------------

def generate_node(
    state: RAGState,
):

    top_documents = (
        state[
            "reranked_documents"
        ][:GENERATION_TOP_K]
    )

    context = format_context(
        top_documents
    )

    prompt = RAG_PROMPT.invoke(
        {
            "context":
                context,

            "question":
                state["question"],
        }
    )

    response = llm.invoke(
        prompt
    )

    sources = []

    for document in (
        top_documents
    ):

        metadata = (
            document.metadata
        )

        source_info = {
            "source":
                metadata.get(
                    "source"
                ),

            "file_type":
                metadata.get(
                    "file_type"
                ),

            "chunk_id":
                metadata.get(
                    "chunk_id"
                ),
        }

        if (
            metadata.get(
                "page_label"
            )
            is not None
        ):

            source_info[
                "page"
            ] = metadata.get(
                "page_label"
            )

        sources.append(
            source_info
        )


    updated_history = (
        state[
            "conversation_history"
        ].copy()
    )

    updated_history.append(
        f"User: {state['question']}"
    )

    updated_history.append(
        f"Assistant: {response.content}"
    )

    return {
        "answer":
            response.content,

        "sources":
            sources,

        "conversation_history":
            updated_history,
    }


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

def build_graph():

    graph = StateGraph(
        RAGState
    )

    graph.add_node(
        "route_question",
        route_question_node,
    )

    graph.add_node(
        "direct_response",
        direct_response_node,
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query_node,
    )

    graph.add_node(
        "metadata_filter",
        metadata_filter_node,
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "rerank",
        rerank_node,
    )

    graph.add_node(
        "relevance_guard",
        relevance_guard_node,
    )

    graph.add_node(
        "no_answer",
        no_answer_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )


    # --------------------------------------------------
    # Start
    # --------------------------------------------------

    graph.add_edge(
        START,
        "route_question",
    )


    # --------------------------------------------------
    # Conversation vs RAG
    # --------------------------------------------------

    graph.add_conditional_edges(
        "route_question",
        choose_route,
        {
            "direct":
                "direct_response",

            "rag":
                "rewrite_query",
        },
    )

    graph.add_edge(
        "direct_response",
        END,
    )


    # --------------------------------------------------
    # RAG pipeline
    # --------------------------------------------------

    graph.add_edge(
        "rewrite_query",
        "metadata_filter",
    )

    graph.add_edge(
        "metadata_filter",
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "rerank",
    )

    graph.add_edge(
        "rerank",
        "relevance_guard",
    )


    # --------------------------------------------------
    # Relevant vs irrelevant
    # --------------------------------------------------

    graph.add_conditional_edges(
        "relevance_guard",
        choose_relevance,
        {
            "relevant":
                "generate",

            "irrelevant":
                "no_answer",
        },
    )

    graph.add_edge(
        "generate",
        END,
    )

    graph.add_edge(
        "no_answer",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer
    )


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    app = build_graph()

    config = {
        "configurable": {
            "thread_id":
                "day4-config-test"
        }
    }

    initial_state = {
        "question":
            "What does NovaSearch do?",

        "standalone_question":
            "",

        "conversation_history":
            [],

        "retrieved_documents":
            [],

        "reranked_documents":
            [],

        "reranker_scores":
            [],

        "answer":
            "",

        "sources":
            [],

        "route":
            "",

        "metadata_filter":
            {},

        "retrieval_relevant":
            False,
    }

    result = app.invoke(
        initial_state,
        config=config,
    )


    print("\nQuestion:")
    print(
        result["question"]
    )

    print("\nRoute:")
    print(
        result["route"]
    )

    print(
        "\nStandalone Question:"
    )

    print(
        result[
            "standalone_question"
        ]
    )

    print(
        "\nMetadata Filter:"
    )

    print(
        result[
            "metadata_filter"
        ]
    )

    print(
        "\nRetrieved Candidates:"
    )

    for index, document in enumerate(
        result[
            "retrieved_documents"
        ],
        start=1,
    ):

        print(
            index,
            document.metadata.get(
                "source"
            ),
            "chunk",
            document.metadata.get(
                "chunk_id"
            ),
        )

    print(
        "\nReranker Scores:"
    )

    print(
        result[
            "reranker_scores"
        ]
    )

    if result[
        "reranker_scores"
    ]:

        print(
            "\nTop Reranker Score:"
        )

        print(
            result[
                "reranker_scores"
            ][0]
        )

    print(
        "\nRetrieval Relevant:"
    )

    print(
        result[
            "retrieval_relevant"
        ]
    )

    print("\nAnswer:")

    print(
        result["answer"]
    )

    print("\nSources:")

    print(
        result["sources"]
    )

    print(
        "\nConversation History:"
    )

    for message in result[
        "conversation_history"
    ]:

        print(
            message
        )