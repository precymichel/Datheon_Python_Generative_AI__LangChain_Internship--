"""Pydantic contracts for strictly validated invoice and resume JSON."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Address(StrictModel):
    line1: str
    city: str | None = None
    region: str | None = None
    postal_code: str | None = None
    country: str | None = None


class Party(StrictModel):
    name: str
    email: str | None = None
    phone: str | None = None
    tax_id: str | None = None
    address: Address | None = None


class LineItem(StrictModel):
    description: str
    quantity: float = Field(gt=0)
    unit: str | None = None
    unit_price: float
    amount: float
    sku: str | None = None

    @field_validator("amount")
    @classmethod
    def amount_matches_qty_price(cls, value: float, info) -> float:
        quantity = info.data.get("quantity")
        unit_price = info.data.get("unit_price")
        if quantity is not None and unit_price is not None:
            expected = round(float(quantity) * float(unit_price), 2)
            if abs(expected - round(value, 2)) > 0.05:
                raise ValueError(
                    f"line amount {value} does not match quantity*unit_price {expected}"
                )
        return value


class InvoiceDocument(StrictModel):
    document_type: Literal["invoice"] = "invoice"
    invoice_number: str
    issue_date: str = Field(description="ISO date YYYY-MM-DD")
    due_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD")
    currency: str = Field(min_length=3, max_length=3)
    purchase_order: str | None = None
    vendor: Party
    bill_to: Party
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float
    tax_total: float = 0
    shipping: float = 0
    total: float
    payment_terms: str | None = None
    notes: str | None = None
    bank_details: str | None = None

    @field_validator("currency")
    @classmethod
    def currency_upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("total")
    @classmethod
    def total_is_consistent(cls, value: float, info) -> float:
        subtotal = info.data.get("subtotal")
        tax_total = info.data.get("tax_total", 0) or 0
        shipping = info.data.get("shipping", 0) or 0
        if subtotal is not None:
            expected = round(float(subtotal) + float(tax_total) + float(shipping), 2)
            if abs(expected - round(value, 2)) > 0.05:
                raise ValueError(
                    f"total {value} does not match subtotal+tax+shipping {expected}"
                )
        return value


class SkillSet(StrictModel):
    technical: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)


class ExperienceRole(StrictModel):
    company: str
    title: str
    location: str | None = None
    start_date: str
    end_date: str | None = Field(default=None, description="ISO date or 'present'")
    highlights: list[str] = Field(default_factory=list)


class EducationEntry(StrictModel):
    institution: str
    credential: str
    field: str | None = None
    end_year: int | None = None
    notes: str | None = None


class Certification(StrictModel):
    name: str
    issuer: str | None = None
    year: int | None = None


class ContactInfo(StrictModel):
    full_name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None


class ResumeDocument(StrictModel):
    document_type: Literal["resume"] = "resume"
    contact: ContactInfo
    summary: str
    skills: SkillSet
    experience: list[ExperienceRole] = Field(min_length=1)
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)


DOCUMENT_SCHEMAS: dict[str, type[InvoiceDocument] | type[ResumeDocument]] = {
    "invoice": InvoiceDocument,
    "resume": ResumeDocument,
}
