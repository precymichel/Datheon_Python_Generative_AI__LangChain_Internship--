from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a careful document question-answering assistant.
Answer the user's question using ONLY the supplied document context.
Do not use outside knowledge or make unsupported assumptions.
If the context does not contain the answer, say:
"I couldn't find that information in the uploaded documents."
When useful, cite the provided source labels inline, for example [report.pdf, page 3].
Be concise, clear, and accurate.

DOCUMENT CONTEXT:
{context}
"""


def load_documents(file_paths: List[str]):
    documents = []
    for file_path in file_paths:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix == ".txt":
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue

        for doc in loaded:
            doc.metadata["source_file"] = path.name
            # PyPDFLoader uses zero-based page metadata. Display page numbers as one-based.
            if suffix == ".pdf" and isinstance(doc.metadata.get("page"), int):
                doc.metadata["page_number"] = doc.metadata["page"] + 1
            else:
                doc.metadata["page_number"] = None
        documents.extend(loaded)

    if not documents:
        raise ValueError("No readable PDF or TXT content was found.")
    return documents


def build_vector_store(file_paths: List[str], chunk_size: int = 900, chunk_overlap: int = 150):
    documents = load_documents(file_paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, max(0, chunk_size - 1)),
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The uploaded documents contain no extractable text.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # In-memory Chroma store: each app session can rebuild it from uploaded files.
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="document_qa_collection",
    )


def _source_label(doc) -> str:
    file_name = doc.metadata.get("source_file") or Path(doc.metadata.get("source", "document")).name
    page_number = doc.metadata.get("page_number")
    if page_number is not None:
        return f"{file_name}, page {page_number}"
    return file_name


def answer_question(
    vector_store,
    question: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    top_k: int = 4,
) -> Dict[str, Any]:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return {
            "answer": "I couldn't find that information in the uploaded documents.",
            "sources": [],
        }

    context_parts = []
    sources = []
    seen = set()
    for doc in relevant_docs:
        label = _source_label(doc)
        context_parts.append(f"[SOURCE: {label}]\n{doc.page_content}")
        key = (label, doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source_file", "Document"),
                "page_label": f"Page {doc.metadata['page_number']}" if doc.metadata.get("page_number") else "Text file",
                "snippet": doc.page_content[:1200],
            })

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nAnswer using only the document context above."),
    ])
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": "\n\n---\n\n".join(context_parts),
        "question": question,
    })
    answer = response.content
    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in answer
        )
    return {"answer": str(answer), "sources": sources}
