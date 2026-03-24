from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"


def get_rag_chain():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        retriever=retriever,
    )
