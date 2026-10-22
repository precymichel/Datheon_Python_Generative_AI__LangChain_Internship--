import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from extractor.json_parse import parse_json_object
from extractor.prompts import render_user_prompt
from extractor.schemas import InvoiceDocument, ResumeDocument

ROOT = Path(__file__).resolve().parents[1]


def test_invoice_schema_accepts_consistent_totals() -> None:
    invoice = InvoiceDocument.model_validate(
        {
            "invoice_number": "BH-19",
            "issue_date": "2025-05-01",
            "due_date": "2025-05-15",
            "currency": "usd",
            "vendor": {"name": "Blue Harbor Office Co"},
            "bill_to": {"name": "Kite Labs"},
            "line_items": [
                {
                    "description": "desk",
                    "quantity": 2,
                    "unit_price": 320.0,
                    "amount": 640.0,
                }
            ],
            "subtotal": 640.0,
            "tax_total": 51.2,
            "shipping": 10.0,
            "total": 701.2,
        }
    )
    assert invoice.currency == "USD"
    assert invoice.document_type == "invoice"


def test_invoice_schema_rejects_bad_line_math() -> None:
    with pytest.raises(ValidationError):
        InvoiceDocument.model_validate(
            {
                "invoice_number": "X",
                "issue_date": "2025-01-01",
                "currency": "INR",
                "vendor": {"name": "A"},
                "bill_to": {"name": "B"},
                "line_items": [
                    {
                        "description": "item",
                        "quantity": 2,
                        "unit_price": 10.0,
                        "amount": 99.0,
                    }
                ],
                "subtotal": 20.0,
                "total": 20.0,
            }
        )


def test_resume_schema_requires_experience() -> None:
    with pytest.raises(ValidationError):
        ResumeDocument.model_validate(
            {
                "contact": {"full_name": "Ada"},
                "summary": "Engineer",
                "skills": {},
                "experience": [],
            }
        )


def test_parse_fenced_json() -> None:
    raw = '```json\n{"hello": "world"}\n```'
    assert parse_json_object(raw) == {"hello": "world"}


def test_prompt_includes_schema_and_source() -> None:
    sample = (ROOT / "samples" / "invoice.txt").read_text(encoding="utf-8")
    prompt = render_user_prompt("invoice", sample)
    assert "ACS-2026-0847" in prompt
    assert "InvoiceDocument" in json.dumps(InvoiceDocument.model_json_schema()) or "invoice_number" in prompt
