from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"


def ingest():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
    print(f"Ingested {len(chunks)} chunks into ChromaDB.")


if __name__ == "__main__":
    ingest()
