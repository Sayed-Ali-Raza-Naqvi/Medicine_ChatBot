from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings

from typing import List


def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()

    return documents


def filter_docs(docs: List[Document]) -> List[Document]:
    filtered_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get('source')
        filtered_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    
    return filtered_docs


def chunk_documents(filtered_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )

    chunked_docs = text_splitter.split_documents(filtered_docs)
    
    return chunked_docs


def download_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return embeddings


