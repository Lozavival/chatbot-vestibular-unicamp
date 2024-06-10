from typing import List

import bs4
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if not load_dotenv():
    raise Exception("Failed to load env variables")


def get_documents(url: str, strainer: bs4.SoupStrainer) -> List[Document]:
    """
    Retrieve and split documents from a given URL.

    Args:
        url (str): The URL of the web page to load documents from.
        strainer (bs4.SoupStrainer): A BeautifulSoup strainer to parse specific parts of the HTML.

    Returns:
        List[Document]: A list of split Document objects.
    """
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": strainer},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def create_embeddings_vectorstore(docs: List[Document]) -> Chroma:
    """
    Create a Chroma vector store from a list of documents.

    Args:
        docs (List[Document]): A list of Document objects to be embedded.

    Returns:
        Chroma: A Chroma vector store containing the document embeddings.
    """
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=HuggingFaceEmbeddings(),
        persist_directory="./data"
    )
    return vectorstore


def create_rag_chain(vectorstore: Chroma | None = None):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.

    Args:
        vectorstore (Chroma | None): An optional Chroma vector store. If not provided, it will be loaded from disk.

    Returns:
        A RAG chain combining document retrieval and question-answering capabilities.
    """
    if not vectorstore:  # load vectorstore from disk
        vectorstore = Chroma(
            persist_directory="./data",
            embedding_function=HuggingFaceEmbeddings()
        )
    retriever = vectorstore.as_retriever(search_type="similarity")

    llm = ChatGroq(temperature=0, model="llama3-70b-8192")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise. Assume all questions "
        "are related to the VU 2024. Always answer in Portuguese. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def chatbot(user_query: str) -> str:
    """
    Process a user query using the RAG chain and return the response.

    Args:
        user_query (str): The user's query to be answered.

    Returns:
        str: The generated response to the user's query.
    """
    response = chatbot.rag_chain.invoke({"input": user_query})
    return response.get("answer")


if not hasattr(chatbot, "rag_chain"):  # emulate C static variable
    chatbot.rag_chain = create_rag_chain()
    print("Criando")
