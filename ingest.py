from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
# replace this:
#from langchain_openai import OpenAIEmbeddings
# with this:
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

PDF_PATH = "data/sample.pdf"
CHROMA_PATH = "chroma_db"


def ingest():
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"Loaded {len(docs)} pages.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    Chroma.from_documents(
        documents=chunks,
       embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=CHROMA_PATH,
    )
    print("Embeddings stored in ChromaDB.")


if __name__ == "__main__":
    ingest()
