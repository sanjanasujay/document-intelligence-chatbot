import streamlit as st
from rag_chain import get_rag_chain

st.set_page_config(page_title="Document Intelligence Chatbot", page_icon="🤖")
st.title("🤖 Document Intelligence Chatbot")

@st.cache_resource
def load_chain():
    return get_rag_chain()

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"query": prompt})
            answer = response["result"]
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
