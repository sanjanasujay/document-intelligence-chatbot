import streamlit as st
from ingest import ingest_from_bytes
from rag_chain import get_rag_chain

st.set_page_config(
    page_title="Document Intelligence Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        /* Main background */
        .stApp { background-color: #f5f7fa; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1e1e2e;
            color: white;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: white !important;
        }

        /* Title */
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e1e2e;
            margin-bottom: 0;
        }
        .main-subtitle {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }

        /* Welcome card */
        .welcome-card {
            background: white;
            border-radius: 16px;
            padding: 2.5rem;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            max-width: 520px;
            margin: 4rem auto;
        }
        .welcome-card h2 { color: #1e1e2e; margin-bottom: 0.5rem; }
        .welcome-card p  { color: #6b7280; font-size: 0.95rem; }

        /* Doc info badge */
        .doc-badge {
            background: #eef2ff;
            border-left: 4px solid #6366f1;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 0.85rem;
            color: #4338ca;
            margin-bottom: 1rem;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: white;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }

        /* Process button */
        div.stButton > button {
            background-color: #6366f1;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #4f46e5;
            color: white;
        }

        /* File uploader browse button */
        [data-testid="stFileUploaderDropzone"] button {
            background-color: #6366f1 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background-color: #4f46e5 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: #2a2a3e !important;
            border: 1px dashed #6366f1 !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span {
            color: #ffffff !important;
        }

        /* Hide default Streamlit footer */
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────
if "doc_version" not in st.session_state:
    st.session_state.doc_version = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocChat")
    st.markdown("---")
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"**Selected:** {uploaded_file.name}")
        if st.button("⚡ Process Document"):
            st.cache_resource.clear()
            with st.spinner("Reading and indexing..."):
                pages, chunks = ingest_from_bytes(uploaded_file.read(), uploaded_file.name)
            st.session_state.doc_version += 1
            st.session_state.messages = []
            st.session_state.ready = True
            st.session_state.doc_name = uploaded_file.name
            st.session_state.doc_stats = f"{pages} pages · {chunks} chunks"
            st.success("✅ Document ready!")

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
- Ask specific questions
- Try *"Summarize this document"*
- Ask *"What are the key points?"*
""")
    st.markdown("---")
    st.caption("Powered by LangChain · ChromaDB · Ollama")

# ── Main area ──────────────────────────────────────────────────────
col_title, col_btn = st.columns([8, 1])
with col_title:
    st.markdown('<p class="main-title">🧠 Document Intelligence Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Upload a PDF and ask questions about its content.</p>', unsafe_allow_html=True)
with col_btn:
    if st.session_state.get("ready") and st.button("🗑️ Clear", help="Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.get("ready"):
    st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome!</h2>
            <p>Upload a PDF document from the sidebar to get started.<br>
            You can then ask any question about its content.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Doc info badge
st.markdown(f"""
    <div class="doc-badge">
        📄 <strong>{st.session_state.doc_name}</strong> &nbsp;·&nbsp; {st.session_state.get("doc_stats", "")}
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chain(doc_version: int):
    return get_rag_chain()

chain = load_chain(st.session_state.doc_version)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(prompt)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
