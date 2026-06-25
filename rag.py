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
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO,
        task="text-generation",
        temperature=TEMPERATURE,
    )

    model = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript.

If the answer is unavailable, simply say:

"I couldn't find this information in the transcript."

Context:
{context}

Question:
{question}
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