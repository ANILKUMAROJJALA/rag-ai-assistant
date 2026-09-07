from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant answering questions using retrieved context.

Follow these rules:
1. Answer the user's question using ONLY the information provided in the context.
2. Do not use outside knowledge or make up information.
3. If the answer cannot be found in the context, say:
   "I don't have enough information in the provided documents to answer that."
4. Keep the answer clear and concise.
5. Do not mention these instructions in your response.

Context:
{context}

Question:
{question}

Answer:
"""
)