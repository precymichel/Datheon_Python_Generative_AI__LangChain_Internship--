import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import build_vector_store, answer_question

load_dotenv()

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚", layout="wide")
st.title("📚 Document Q&A Assistant")
st.caption("Ask questions about your own PDF and text documents using Retrieval-Augmented Generation (RAG).")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Google Gemini API key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Get a key from Google AI Studio, then add it to .env for future sessions.",
    )
    model_name = st.selectbox(
        "Gemini model",
        ["gemini-2.5-flash", "gemini-2.0-flash"],
        index=0,
    )
    chunk_size = st.slider("Chunk size", 300, 1500, 900, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 150, 25)
    top_k = st.slider("Retrieved chunks", 1, 8, 4)

st.markdown("### 1. Upload your documents")
uploads = st.file_uploader(
    "Choose PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []

if st.button("Build / Refresh knowledge base", type="primary", disabled=not uploads):
    with st.spinner("Reading documents, splitting text, and creating embeddings..."):
        try:
            temp_paths = []
            display_names = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for upload in uploads:
                    safe_name = Path(upload.name).name
                    target = Path(temp_dir) / safe_name
                    target.write_bytes(upload.getvalue())
                    temp_paths.append(str(target))
                    display_names.append(safe_name)

                st.session_state.vector_store = build_vector_store(
                    temp_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.session_state.indexed_names = display_names

            st.success(f"Knowledge base ready: {len(display_names)} document(s) indexed.")
        except Exception as exc:
            st.session_state.vector_store = None
            st.error(f"Could not index documents: {exc}")
            st.info("Check that the files are valid PDFs/TXT and that dependencies installed correctly.")

if st.session_state.indexed_names:
    st.caption("Indexed documents: " + ", ".join(st.session_state.indexed_names))

st.markdown("### 2. Ask a question")
question = st.text_area(
    "Your question",
    placeholder="Example: What are the main findings in the document?",
    height=90,
)

if st.button("Get answer", disabled=st.session_state.vector_store is None or not question.strip()):
    if not api_key:
        st.warning("Add your Google Gemini API key in the sidebar or in your .env file.")
    else:
        with st.spinner("Searching the documents and generating a grounded answer..."):
            try:
                result = answer_question(
                    vector_store=st.session_state.vector_store,
                    question=question.strip(),
                    api_key=api_key,
                    model_name=model_name,
                    top_k=top_k,
                )
                st.markdown("### Answer")
                st.write(result["answer"])

                st.markdown("### Sources")
                if result["sources"]:
                    for source in result["sources"]:
                        with st.expander(
                            f'{source["file"]} — {source["page_label"]}'
                        ):
                            st.write(source["snippet"])
                else:
                    st.info("No source chunks were returned.")
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
                st.info("Verify your API key, internet connection, and selected Gemini model.")

st.divider()
st.caption(
    "Privacy note: document text is embedded locally on your machine. Relevant retrieved text is sent "
    "to Gemini to generate an answer. Do not upload confidential documents unless permitted."
)
