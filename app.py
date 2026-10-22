
"""Structured LLM Prompt Pipeline & JSON Extractor."""

import json
from pathlib import Path

import streamlit as st
from extractor.pipeline import ExtractionPipeline

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"

st.set_page_config(
    page_title="Structured LLM Prompt Pipeline & JSON Extractor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- COMPLETE UI STYLING --------------------
st.markdown(
    """
    <style>
    :root {
        --page: #F4F5FF;
        --card: #FFFFFF;
        --ink: #24234F;
        --muted: #77769D;
        --purple: #635BFF;
        --border: #E5E7F8;
    }

    .stApp {
        background: #F4F5FF !important;
        color: var(--ink);
    }

    /* Keep Streamlit's toolbar separate from the app's own header. */
    [data-testid="stHeader"] {
        background: #F4F5FF;
        height: 3rem;
        z-index: 100;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        width: 100%;
        max-width: 1250px;
        /* Extra top space prevents the fixed Streamlit toolbar clipping the brand card. */
        padding: 3.75rem 2rem 3rem;
        margin-left: auto;
        margin-right: auto;
        box-sizing: border-box;
    }

    /* HEADER: separate white card with clear text */
    .brand-row {
        display: grid;
        grid-template-columns: 48px minmax(0, 1fr) auto;
        align-items: center;
        column-gap: 16px;
        row-gap: 8px;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        padding: 16px 20px;
        margin: 4px 0 22px 0;
        background: #FFFFFF;
        border: 1px solid #E5E7F8;
        border-radius: 18px;
        box-shadow: 0 6px 22px rgba(50,45,120,.07);
        animation: fadeIn .45s ease both;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        color: #FFFFFF !important;
        background: linear-gradient(135deg,#635BFF,#9B5CFF);
        font-size: 26px;
        box-shadow: 0 5px 14px rgba(99,91,255,.22);
    }

    .brand-row > div:nth-child(2) {
        min-width: 0;
        width: 100%;
    }

    .brand-name {
        color: #24234F !important;
        font-size: clamp(15px, 1.5vw, 19px);
        font-weight: 800;
        line-height: 1.35;
        margin: 0;
        opacity: 1 !important;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    .brand-subtitle {
        color: #77769D !important;
        font-size: 13px;
        line-height: 1.5;
        margin-top: 4px;
        opacity: 1 !important;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    .ready-pill {
        justify-self: end;
        align-self: center;
        max-width: 100%;
        box-sizing: border-box;
        padding: 6px 12px;
        border-radius: 999px;
        background: #D8F7E8;
        color: #087A45 !important;
        font-size: 12px;
        font-weight: 750;
        white-space: nowrap;
    }

    .hero-title {
        overflow-wrap: anywhere;
        word-break: normal;
    }

    /* MAIN TITLE CARD */
    .hero-card {
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        background: #FFFFFF;
        border: 1px solid #E8E9F7;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 16px;
        box-shadow: 0 7px 24px rgba(50,45,120,.045);
        animation: fadeIn .55s ease both;
    }

    .hero-title {
        font-size: clamp(22px, 2.5vw, 30px);
        font-weight: 800;
        letter-spacing: -.7px;
        line-height: 1.35;
        color: #24234F !important;
        margin: 0 0 10px;
    }

    .hero-description {
        color: #77769D !important;
        font-size: 14px;
        line-height: 1.8;
        margin: 0 0 14px;
    }

    .tech-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .tech-tag {
        font-size: 11px;
        padding: 6px 11px;
        border-radius: 999px;
        font-weight: 700;
    }

    .tag-blue {
        background: #E2EEFF;
        color: #2463D4;
    }

    .tag-purple {
        background: #F0E5FF;
        color: #7C3AED;
    }

    .tag-green {
        background: #DCF8E8;
        color: #078447;
    }

    /* FEATURE CARDS */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2,minmax(0,1fr));
        gap: 13px;
        margin-bottom: 25px;
    }

    .feature-card {
        background: #FFFFFF;
        border: 1px solid #EAEBF8;
        border-radius: 17px;
        padding: 20px;
        min-height: 110px;
        box-shadow: 0 5px 18px rgba(50,45,120,.04);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 26px rgba(99,91,255,.10);
    }

    .feature-icon {
        color: #635BFF;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .feature-name {
        color: #24234F !important;
        font-size: 15px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .feature-description {
        color: #77769D !important;
        font-size: 13px;
        line-height: 1.6;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7F8;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #39318F !important;
    }

    /* GENERAL TEXT */
    h1, h2, h3 {
        color: #292751 !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #77769D !important;
    }

    /* INPUTS */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #FFFFFF !important;
        border: 1px solid #E1E3F2 !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="textarea"] textarea {
        background: #FFFFFF !important;
        color: #24234F !important;
        border-radius: 12px !important;
        line-height: 1.7;
        caret-color: #635BFF !important;
    }

    div[data-baseweb="input"] input {
        background: #FFFFFF !important;
        color: #24234F !important;
    }

    div[data-baseweb="textarea"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #8B83FF !important;
        box-shadow: 0 0 0 3px rgba(99,91,255,.12) !important;
    }

    /* GRADIENT BUTTONS */
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 12px !important;
        min-height: 45px;
        font-weight: 750 !important;
        transition: transform .2s ease, box-shadow .2s ease;
    }

    div.stButton > button[kind="primary"],
    div.stDownloadButton > button {
        color: #FFFFFF !important;
        background: linear-gradient(110deg,#635BFF,#8B5CF6) !important;
        border: 0 !important;
        box-shadow: 0 7px 19px rgba(99,91,255,.22);
    }

    div.stButton > button[kind="primary"] p,
    div.stDownloadButton > button p {
        color: #FFFFFF !important;
        font-weight: 750 !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 11px 25px rgba(99,91,255,.30);
    }

    /* RESULTS METRICS */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7F8;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 5px 16px rgba(50,45,120,.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #77769D !important;
    }

    div[data-testid="stMetricValue"] {
        color: #5B50E6 !important;
        font-weight: 800;
    }

    div[data-testid="stJson"] {
        background: #FFFFFF;
        border: 1px solid #E5E7F8;
        border-radius: 14px;
        padding: 12px;
    }

    button[data-baseweb="tab"] {
        color: #77769D !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #635BFF !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* FOOTER */
    .footer {
        border-top: 1px solid #DFE2F2;
        margin-top: 28px;
        padding-top: 15px;
        color: #77769D !important;
        font-size: 12px;
        line-height: 1.8;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 650px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .block-container {
            padding: 3.5rem 1rem 2rem;
        }

        .brand-row {
            grid-template-columns: 40px minmax(0, 1fr);
            align-items: start;
            column-gap: 10px;
            row-gap: 8px;
            padding: 12px;
            margin-top: 2px;
            margin-bottom: 16px;
        }

        .brand-icon {
            width: 40px;
            height: 40px;
            font-size: 22px;
            grid-column: 1;
            grid-row: 1;
        }

        .brand-row > div:nth-child(2) {
            grid-column: 2;
            grid-row: 1;
        }

        .brand-name {
            font-size: 15px;
            line-height: 1.35;
        }

        .brand-subtitle {
            font-size: 11px;
            line-height: 1.45;
        }

        .ready-pill {
            grid-column: 2;
            grid-row: 2;
            justify-self: start;
            font-size: 10px;
            padding: 5px 8px;
        }

        .hero-title {
            font-size: clamp(21px, 6vw, 28px);
            line-height: 1.3;
            letter-spacing: -0.35px;
        }

        .hero-card {
            padding: 18px;
        }

        .feature-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- HEADER --------------------
st.markdown(
    """
    <div class="brand-row">
        <div class="brand-icon">✧</div>
        <div>
            <div class="brand-name">AI Document Intelligence</div>
            <div class="brand-subtitle">
                Smart extraction · Structured output · Validation
            </div>
        </div>
        <div class="ready-pill">● AI Ready</div>
    </div>

    <div class="hero-card">
        <div class="hero-title">
            Structured LLM Prompt Pipeline &amp; JSON Extractor
        </div>
        <div class="hero-description">
            Transform messy invoices and resumes into clean,
            validated JSON with Google Gemini.
        </div>
        <div class="tech-tags">
            <span class="tech-tag tag-blue">Google Gemini</span>
            <span class="tech-tag tag-purple">Pydantic</span>
            <span class="tech-tag tag-green">JSON Validation</span>
        </div>
    </div>

    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">▤</div>
            <div class="feature-name">Document input</div>
            <div class="feature-description">
                Paste, upload or use a sample
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon" style="color:#0891B2;">{ }</div>
            <div class="feature-name">JSON output</div>
            <div class="feature-description">
                Validated, formatted and downloadable
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## Extraction settings")

    document_type = st.selectbox(
        "Document type",
        ["invoice", "resume"],
        format_func=lambda value: value.title(),
    )

    input_mode = st.radio(
        "Input method",
        ["Paste text", "Upload .txt", "Use sample"],
    )

    st.divider()
    st.markdown("### Processing pipeline")
    st.caption("1. Prompt construction")
    st.caption("2. Gemini generation")
    st.caption("3. JSON parsing")
    st.caption("4. Pydantic validation")
    st.caption("5. Repair and retry")

# -------------------- DOCUMENT INPUT --------------------
st.markdown("## Document input")
st.caption("Provide the text you want to convert into structured JSON.")

source_text = ""

if input_mode == "Paste text":
    source_text = st.text_area(
        "Document text",
        height=250,
        placeholder="Paste invoice details or resume content here...",
        key="document_text",
    )

elif input_mode == "Upload .txt":
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["txt"],
    )

    if uploaded_file is not None:
        source_text = uploaded_file.getvalue().decode(
            "utf-8", errors="replace"
        )

        st.text_area(
            "Document preview",
            value=source_text,
            height=220,
            disabled=True,
        )

else:
    sample_path = SAMPLES_DIR / f"{document_type}.txt"

    if sample_path.exists():
        source_text = sample_path.read_text(encoding="utf-8")

        st.text_area(
            "Sample document preview",
            value=source_text,
            height=250,
            disabled=True,
        )
        st.info(f"Using the sample {document_type}.")
    else:
        st.error(f"Sample file not found: {sample_path.name}")

# -------------------- RUN EXTRACTION --------------------
if st.button(
    "✦  Extract with Gemini",
    type="primary",
    use_container_width=True,
):
    st.session_state.pop("extraction_result", None)

    if not source_text.strip():
        st.warning("Please provide document text before extracting.")
    else:
        try:
            with st.spinner(
                "Gemini is extracting fields and validating JSON..."
            ):
                pipeline = ExtractionPipeline()
                result = pipeline.extract(
                    source_text=source_text,
                    document_type=document_type,
                )

            st.session_state["extraction_result"] = {
                "data": result.data,
                "document_type": result.document_type,
                "attempts": result.attempts,
                "model": result.model,
                "raw_responses": result.raw_responses,
            }

        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            st.info(
                "Check your API key, model configuration, "
                "network connection, and VS Code terminal."
            )

# -------------------- EXTRACTION RESULTS --------------------
saved_result = st.session_state.get("extraction_result")

if saved_result:
    st.divider()
    st.markdown("## Extraction results")
    st.success("Extraction completed and passed schema validation.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Document type",
        saved_result["document_type"].title(),
    )
    col2.metric("Attempts used", saved_result["attempts"])
    col3.metric("Validation", "Passed")

    st.caption(
        f"Model: {saved_result['model'] or 'Configured Gemini model'}"
    )

    json_text = json.dumps(
        saved_result["data"],
        indent=2,
        ensure_ascii=False,
    )

    json_tab, raw_tab = st.tabs(
        ["Validated JSON", "Raw model responses"]
    )

    with json_tab:
        st.json(saved_result["data"], expanded=True)

        st.download_button(
            "Download validated JSON",
            data=json_text,
            file_name=(
                f"{saved_result['document_type']}_extracted.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

    with raw_tab:
        for index, raw in enumerate(
            saved_result["raw_responses"], start=1
        ):
            with st.expander(f"Attempt {index}"):
                st.code(raw, language="json")

# -------------------- FOOTER --------------------
st.markdown(
    """
    <div class="footer">
        Structured LLM Prompt Pipeline &amp; JSON Extractor
        · Review extracted information before using it.
        AI-generated results may contain errors.
    </div>
    """,
    unsafe_allow_html=True,
)
