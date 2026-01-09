import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
embedding = HuggingFaceEmbeddings()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0  # Fixed typo
)

def process_document(file_path):
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_path}")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/vector_db"
    )
    return 0

def answer_question(user_question):
    vectordb = Chroma(
        persist_directory=f"{working_dir}/vector_db",
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever()

    # Simple prompt for RAG
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If you don't know the answer based on the context, say so.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_question})
    return response["answer"]
