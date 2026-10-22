# Document Q&A Assistant with RAG & Vector Embeddings

 Retrieval-Augmented Generation (RAG) application that lets users upload PDF/TXT documents and ask questions grounded in those documents.

## Features

- Upload multiple PDF and TXT documents.
- Extract PDF text with page metadata.
- Split content using `RecursiveCharacterTextSplitter`.
- Create local embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Store vectors in ChromaDB for semantic retrieval.
- Use Google Gemini to answer questions from retrieved context only.
- Display source file and page information.
- Streamlit web interface with adjustable chunk size, overlap, retrieval count, and model.

## Project structure

```text
document_qa_rag_assistant/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or 3.11 recommended.
- Internet connection for installing packages, downloading the embedding model the first time, and calling Gemini.
- A Google Gemini API key from Google AI Studio.

## Run on Windows (VS Code)

Open the project folder in VS Code. In the terminal, run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv` with a supported Python installation.

Create your local environment file:

```bat
copy .env.example .env
```

Open `.env` and replace `your_gemini_api_key_here` with your actual key. Keep `.env` private and never commit it to GitHub.

Start the app:

```bat
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## How to use

1. Upload one or more PDF or TXT files.
2. Select **Build / Refresh knowledge base** and wait for indexing to finish.
3. Enter a question that can be answered from the uploaded documents.
4. Select **Get answer**.
5. Review the answer and expand source entries to inspect the retrieved text.

## How the RAG pipeline works

1. **Load:** `PyPDFLoader` reads PDF pages; `TextLoader` reads TXT files.
2. **Split:** `RecursiveCharacterTextSplitter` creates overlapping chunks.
3. **Embed:** a local Sentence Transformers model converts chunks into vectors.
4. **Store:** ChromaDB stores vectors for semantic similarity search.
5. **Retrieve:** the retriever finds the most relevant chunks for a question.
6. **Generate:** Gemini receives only the retrieved context and the question.
7. **Cite:** source file/page labels are added to context and source snippets are shown in the UI.

## Important notes

- This app uses a local in-memory Chroma collection. Rebuild the knowledge base after refreshing the page or starting a new session.
- The first embedding-model download can take some time and requires internet.
- Scanned PDFs that contain only images may need OCR before they can be searched.
- RAG can still produce mistakes. Check the cited source passages for important decisions.
- The retrieved document excerpts are sent to Google Gemini. Do not use confidential data unless you have permission.
- Gemini model availability can vary by account/region. If the selected model is unavailable, try the other option in the sidebar.
- Never commit `.env` or API keys to GitHub.

## Troubleshooting

### `GOOGLE_API_KEY` missing
Enter the key in the sidebar or add it to `.env`, then restart the app.

### ChromaDB or PyTorch install error
Use Python 3.10/3.11 in a fresh virtual environment and rerun `pip install -r requirements.txt`.

### PDF answers are empty
Check whether the PDF contains selectable text. Image-only scanned PDFs need OCR.

### Model not found / API error
Check your Google AI Studio key and the Gemini model selected in the sidebar.

## Suggested GitHub description

`A Streamlit RAG application for PDF/TXT question answering using LangChain, local Sentence Transformers embeddings, ChromaDB, and Google Gemini with source-page references.`
