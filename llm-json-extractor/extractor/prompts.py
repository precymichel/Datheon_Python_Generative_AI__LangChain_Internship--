"""Parameterized prompt templates with few-shot examples."""

from __future__ import annotations

from textwrap import dedent

from extractor.schemas import DOCUMENT_SCHEMAS

SYSTEM_INSTRUCTIONS = dedent(
    """
    You extract structured data from unstructured business documents.
    Return ONLY valid JSON that matches the provided schema. No markdown.
    Rules:
    - Use ISO dates (YYYY-MM-DD) when a day is known; otherwise the first of the month.
    - Currency must be a 3-letter ISO code (INR, USD, EUR).
    - Never invent emails, tax IDs, or bank accounts that are not in the source.
    - If a field is unknown, use null or an empty list as the schema allows.
    - Numbers must be JSON numbers, not strings with currency symbols.
    - For invoices, line amount must equal quantity * unit_price (2 decimal rounding).
    - For invoices, total must equal subtotal + tax_total + shipping.
    """
).strip()

FEW_SHOTS: dict[str, list[dict[str, str]]] = {
    "invoice": [
        {
            "input": (
                "Blue Harbor Office Co / Invoice BH-19 / 1 May 2025 / due 15 May 2025 / USD\n"
                "Bill to: Kite Labs, 9 Market St, Austin\n"
                "2x standing desks @ 320.00 = 640.00 ; 1x chair @ 210.00 = 210.00\n"
                "Subtotal 850.00 tax 68.00 shipping 40.00 total 958.00"
            ),
            "output": """{
  "document_type": "invoice",
  "invoice_number": "BH-19",
  "issue_date": "2025-05-01",
  "due_date": "2025-05-15",
  "currency": "USD",
  "purchase_order": null,
  "vendor": {"name": "Blue Harbor Office Co", "email": null, "phone": null, "tax_id": null, "address": null},
  "bill_to": {"name": "Kite Labs", "email": null, "phone": null, "tax_id": null,
    "address": {"line1": "9 Market St", "city": "Austin", "region": null, "postal_code": null, "country": null}},
  "line_items": [
    {"description": "standing desks", "quantity": 2, "unit": null, "unit_price": 320.0, "amount": 640.0, "sku": null},
    {"description": "chair", "quantity": 1, "unit": null, "unit_price": 210.0, "amount": 210.0, "sku": null}
  ],
  "subtotal": 850.0,
  "tax_total": 68.0,
  "shipping": 40.0,
  "total": 958.0,
  "payment_terms": null,
  "notes": null,
  "bank_details": null
}""",
        }
    ],
    "resume": [
        {
            "input": (
                "Maya Chen — Seattle — maya@example.com — Python, AWS\n"
                "Staff Engineer at Orbit, 2021-present: led payments API.\n"
                "B.S. Computer Science, UW, 2016."
            ),
            "output": """{
  "document_type": "resume",
  "contact": {"full_name": "Maya Chen", "email": "maya@example.com", "phone": null,
    "location": "Seattle", "linkedin": null, "github": null, "website": null},
  "summary": "Staff engineer with Python and AWS experience, currently leading payments API work at Orbit.",
  "skills": {"technical": ["Python"], "tools": ["AWS"], "other": []},
  "experience": [
    {"company": "Orbit", "title": "Staff Engineer", "location": null,
     "start_date": "2021-01-01", "end_date": "present",
     "highlights": ["Led payments API"]}
  ],
  "education": [
    {"institution": "UW", "credential": "B.S.", "field": "Computer Science", "end_year": 2016, "notes": null}
  ],
  "certifications": []
}""",
        }
    ],
}

USER_TEMPLATE = dedent(
    """
    Document type: {document_type}

    JSON schema (informational — obey every required field and type):
    {schema_json}

    Extract from the source document below. Output a single JSON object only.

    --- SOURCE DOCUMENT ---
    {source_text}
    --- END SOURCE DOCUMENT ---
    """
).strip()

REPAIR_TEMPLATE = dedent(
    """
    Your previous JSON failed Pydantic validation.

    Validation errors:
    {errors}

    Invalid JSON you returned:
    {invalid_json}

    Source document (again):
    --- SOURCE DOCUMENT ---
    {source_text}
    --- END SOURCE DOCUMENT ---

    Return a corrected JSON object only. Do not wrap it in markdown.
    """
).strip()


def schema_json_for(document_type: str) -> str:
    model = DOCUMENT_SCHEMAS[document_type]
    return model.model_json_schema()


def render_user_prompt(document_type: str, source_text: str) -> str:
    import json

    return USER_TEMPLATE.format(
        document_type=document_type,
        schema_json=json.dumps(schema_json_for(document_type), indent=2),
        source_text=source_text.strip(),
    )


def render_repair_prompt(source_text: str, invalid_json: str, errors: str) -> str:
    return REPAIR_TEMPLATE.format(
        errors=errors.strip(),
        invalid_json=invalid_json.strip(),
        source_text=source_text.strip(),
    )


def few_shot_turns(document_type: str) -> list[dict[str, str]]:
    return FEW_SHOTS.get(document_type, [])
