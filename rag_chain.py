from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"

PROMPT = ChatPromptTemplate.from_template("""
You are a document intelligence assistant.
Answer the user's question using only the retrieved context below.
If the answer is not in the context, say you could not find it in the document.

Context:
{context}

Question:
{question}
""")


def get_rag_chain():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | Ollama(model="llama3")
        | StrOutputParser()
    )
