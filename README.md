# Structured LLM Prompt Pipeline & JSON Extractor

Python pipeline that turns messy invoice or resume text into **strictly validated JSON** using the **Google Gemini** API, parameterized prompt templates, few-shot examples, and **Pydantic**.

This is Task 1 of a structured LLM / portfolio programme: connect Python to Gemini, engineer prompts, and refuse invalid model output.

## What you get

| Piece | Role |
| --- | --- |
| `extract.py` | Executable CLI |
| `extractor/prompts.py` | Parameterized templates + few-shot turns |
| `extractor/gemini_client.py` | `google-generativeai` JSON mode client |
| `extractor/schemas.py` | Invoice and resume Pydantic contracts |
| `extractor/pipeline.py` | Parse → validate → repair retry loop |
| `samples/` | Unstructured invoice and resume documents |
| `tests/` | Schema and prompt tests (no API key required) |

## Setup

```powershell
cd C:\Users\User\llm-json-extractor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Put a Gemini API key in `.env`:

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.8-flash
```

Create a key at [Google AI Studio](https://aistudio.google.com/apikey).

## Run

Preview the prompt (no API call):

```powershell
python extract.py --type invoice --file samples\invoice.txt --print-prompt
python extract.py --type resume --file samples\resume.txt --print-schema
```

Extract and validate (writes `output/<name>.json` and prints JSON):

```powershell
python extract.py --type invoice --file samples\invoice.txt
python extract.py --type resume --file samples\resume.txt --out output\resume.json
python extract.py --type invoice --text "Invoice 12 ... total 100.00"
```

## How the pipeline works

1. Render a **parameterized** user prompt (`document_type`, JSON schema, source text).
2. Prefix **few-shot** invoice or resume examples.
3. Call Gemini with `response_mime_type=application/json`.
4. Parse the reply (including accidental markdown fences).
5. Validate with Pydantic (`extra` fields forbidden; invoice line math and totals checked).
6. On failure, send the validation errors back and retry (default 2 repairs).

## Tests

```powershell
pip install pytest
pytest -q
```

These tests do not call Gemini.

## Project layout

```
extract.py
extractor/
  schemas.py
  prompts.py
  gemini_client.py
  json_parse.py
  pipeline.py
samples/
  invoice.txt
  resume.txt
tests/
  test_schemas.py
```
