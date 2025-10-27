import os
from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_docs, chunk_documents, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

extracted_data = load_pdf_files("data/")
filtered_docs = filter_docs(extracted_data)
chunked_docs = chunk_documents(filtered_docs)

embeddings = download_embeddings()

index_name = "medical-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

vectorstore.add_documents(chunked_docs)

print("✅ Index created and documents successfully uploaded to Pinecone!")