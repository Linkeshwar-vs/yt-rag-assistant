from transformers.utils import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from config import *

logging.set_verbosity_error()
reranker = CrossEncoder(RERANKER_MODEL)


def format_docs(docs):

    formatted = []

    for i, doc in enumerate(docs, start=1):

        formatted.append(
            f"Transcript Section {i}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


def rerank_documents(question, docs):

    if len(docs) <= FINAL_K:
        return docs

    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for _, doc in ranked_docs[:FINAL_K]
    ]


def build_rag_chain(transcript):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": TOP_K * 2,
            "lambda_mult": 0.7,
        },
    )

    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO,
        task="text-generation",
        temperature=TEMPERATURE,
    )

    model = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        template="""
    You are a Retrieval-Augmented Generation (RAG) assistant.

    Your ONLY source of truth is the transcript context provided below.

    Rules:

    - Use ONLY the transcript.
    - NEVER use outside knowledge.
    - NEVER guess.
    - NEVER hallucinate.
    - If the answer is not present in the transcript, respond exactly:

    "I couldn't find this information in the transcript."

    - When information is spread across multiple transcript sections, combine it into one coherent answer.
    - Prefer detailed explanations over one-line responses.
    - Organize long answers using bullet points when appropriate.
    - Quote or paraphrase relevant transcript information whenever possible.

    Transcript Context:
    --------------------
    {context}
    --------------------

    Question:
    {question}

    Answer:
    """,
        input_variables=["context", "question"],
    )

    def retrieve_and_rerank(question):

        docs = retriever.invoke(question)

        docs = rerank_documents(question, docs)

        return format_docs(docs)

    chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(retrieve_and_rerank),
                "question": RunnablePassthrough(),
            }
        ) | prompt | model | StrOutputParser()
    )

    return chain