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
from config import *
from utils import format_docs


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
            "k": 8,
            "fetch_k": 20,
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

    Use ONLY the transcript context below.

    Instructions:

    - Never use outside knowledge.
    - If the answer is not present, reply exactly:
    "I couldn't find this information in the transcript."
    - Give complete, informative answers.
    - When multiple transcript sections discuss the topic, combine them into one coherent response.
    - If appropriate, organize the answer using bullet points.
    - Do not repeat information unnecessarily.

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

    chain = (
        RunnableParallel(
            {
                "context": retriever
                | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )   | prompt | model | StrOutputParser()
    )
    return chain