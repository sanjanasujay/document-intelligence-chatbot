import streamlit as st
from ingest import ingest_from_bytes
from rag_chain import get_rag_chain

st.set_page_config(page_title="Document Intelligence Chatbot", page_icon="🤖")
st.title("🤖 Document Intelligence Chatbot")

if "doc_version" not in st.session_state:
    st.session_state.doc_version = 0

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Ingesting document..."):
                pages, chunks = ingest_from_bytes(uploaded_file.read(), uploaded_file.name)
            st.session_state.doc_version += 1
            st.session_state.messages = []
            st.session_state.ready = True
            st.success(f"Done! {pages} pages, {chunks} chunks indexed.")

if not st.session_state.get("ready"):
    st.info("Upload and process a PDF from the sidebar to get started.")
    st.stop()

@st.cache_resource
def load_chain(doc_version: int):
    return get_rag_chain()

chain = load_chain(st.session_state.doc_version)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(prompt)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
