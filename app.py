# app.py

import streamlit as st
from PyPDF2 import PdfReader
from agent import add_to_db, ask
import uuid

st.set_page_config(page_title="Research Paper Q&A", layout="wide")

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False


# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("📄 Research Assistant")

    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.success("New conversation started!")


# -------------------------
# PDF → TEXT
# -------------------------
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# -------------------------
# CHUNKING (IMPROVED)
# -------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------
# MAIN TITLE
# -------------------------
st.title("📄 Research Paper Q&A Assistant")

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------
# PROCESS FILES
# -------------------------
if uploaded_files and not st.session_state.db_loaded:
    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            text = extract_text(file)
            chunks = chunk_text(text)

            # DEBUG
            print(f"\nProcessing {file.name}, chunks: {len(chunks)}")

            add_to_db(chunks, source_name=file.name)

    st.session_state.db_loaded = True
    st.success("Documents processed successfully!")


# -------------------------
# SHOW CHAT HISTORY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------
# USER INPUT
# -------------------------
user_input = st.chat_input("Ask a question from your papers...")

if user_input:
    st.chat_message("user").write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        answer, sources = ask(
            user_input,
            thread_id=st.session_state.thread_id
        )

    with st.chat_message("assistant"):
        st.write(answer)

        if sources:
            st.markdown("**Sources:**")
            for s in set(sources):
                st.write(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })