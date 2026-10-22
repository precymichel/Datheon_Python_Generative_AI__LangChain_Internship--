#!/usr/bin/env python3
"""CLI: parse unstructured invoice or resume text into validated JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from extractor.pipeline import ExtractionPipeline, dump_schema_example
from extractor.prompts import render_user_prompt
from extractor.schemas import DOCUMENT_SCHEMAS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract strictly validated JSON from unstructured documents via Gemini."
    )
    parser.add_argument(
        "--type",
        dest="document_type",
        required=True,
        choices=sorted(DOCUMENT_SCHEMAS),
        help="Document schema to enforce.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Path to a .txt / .md source document.")
    source.add_argument("--text", help="Raw document text.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Write JSON here. Default: stdout (and output/<stem>.json when --file is used).",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the rendered prompt and exit without calling Gemini.",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the Pydantic JSON schema and exit.",
    )
    args = parser.parse_args(argv)

    if args.print_schema:
        print(json.dumps(dump_schema_example(args.document_type), indent=2))
        return 0

    source_text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")

    if args.print_prompt:
        print(render_user_prompt(args.document_type, source_text))
        return 0

    try:
        result = ExtractionPipeline().extract(source_text, args.document_type)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(result.data, indent=2, ensure_ascii=False)
    out_path = args.out
    if out_path is None and args.file is not None:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{args.file.stem}.json"

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}  (attempts={result.attempts}, model={result.model})")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
