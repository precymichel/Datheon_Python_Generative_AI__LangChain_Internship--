"""Prompt → Gemini → JSON parse → Pydantic validate, with repair retries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from extractor.json_parse import parse_json_object
from extractor.prompts import (
    SYSTEM_INSTRUCTIONS,
    few_shot_turns,
    render_repair_prompt,
    render_user_prompt,
)
from extractor.schemas import DOCUMENT_SCHEMAS, StrictModel


@dataclass
class ExtractionResult:
    document_type: str
    data: dict[str, Any]
    attempts: int
    raw_responses: list[str] = field(default_factory=list)
    model: str = ""


class ExtractionPipeline:
    def __init__(self, client=None, max_repairs: int = 2) -> None:
        if client is None:
            from extractor.gemini_client import GeminiJsonClient

            client = GeminiJsonClient()
        self.client = client
        self.max_repairs = max_repairs

    def extract(self, source_text: str, document_type: str) -> ExtractionResult:
        document_type = document_type.lower().strip()
        if document_type not in DOCUMENT_SCHEMAS:
            allowed = ", ".join(sorted(DOCUMENT_SCHEMAS))
            raise ValueError(f"unknown document_type {document_type!r}; use {allowed}")
        if not source_text.strip():
            raise ValueError("source_text is empty")

        schema = DOCUMENT_SCHEMAS[document_type]
        contents = self._initial_contents(document_type, source_text)
        raw_responses: list[str] = []
        last_error = "unknown error"

        for attempt in range(1, self.max_repairs + 2):
            raw = self.client.complete(SYSTEM_INSTRUCTIONS, contents)
            raw_responses.append(raw)
            try:
                payload = parse_json_object(raw)
                model = schema.model_validate(payload)
                return ExtractionResult(
                    document_type=document_type,
                    data=model.model_dump(mode="json"),
                    attempts=attempt,
                    raw_responses=raw_responses,
                    model=getattr(self.client.settings, "model", ""),
                )
            except (json.JSONDecodeError, ValueError, ValidationError) as exc:
                last_error = _format_error(exc)
                contents = contents + [
                    {"role": "model", "text": raw or "{}"},
                    {
                        "role": "user",
                        "text": render_repair_prompt(source_text, raw or "", last_error),
                    },
                ]

        raise RuntimeError(
            f"extraction failed after {len(raw_responses)} attempt(s): {last_error}"
        )

    def _initial_contents(self, document_type: str, source_text: str) -> list[dict[str, str]]:
        contents: list[dict[str, str]] = []
        for shot in few_shot_turns(document_type):
            contents.append({"role": "user", "text": shot["input"]})
            contents.append({"role": "model", "text": shot["output"]})
        contents.append(
            {"role": "user", "text": render_user_prompt(document_type, source_text)}
        )
        return contents


def _format_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return exc.json()
    return str(exc)


def dump_schema_example(document_type: str) -> dict[str, Any]:
    schema: type[StrictModel] = DOCUMENT_SCHEMAS[document_type]
    return schema.model_json_schema()
