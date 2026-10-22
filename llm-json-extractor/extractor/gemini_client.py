
"""Google Gemini client using the current Google GenAI SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types


class MissingApiKeyError(RuntimeError):
    """Raised when the Gemini API key is missing."""


@dataclass
class GeminiSettings:
    api_key: str
    model: str = "gemini-2.0-flash"


def load_settings() -> GeminiSettings:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key.startswith("your_"):
        raise MissingApiKeyError(
            "Set GEMINI_API_KEY in your .env file. "
            "Create a key at https://aistudio.google.com/apikey"
        )

    model = (
        os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        or "gemini-2.0-flash"
    )

    return GeminiSettings(api_key=api_key, model=model)


class GeminiJsonClient:
    """Generate JSON responses through the Google GenAI SDK."""

    def __init__(
        self,
        settings: GeminiSettings | None = None,
        system_instruction: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()

        self.client = genai.Client(api_key=self.settings.api_key)

        self.system_instruction = system_instruction

        self.generation_config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )

    def complete(
        self,
        system: str,
        contents: list[dict[str, str]],
    ) -> str:
        """Send conversation history and return generated text."""

        conversation = []

        for turn in contents:
            role = "user" if turn["role"] == "user" else "model"

            conversation.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn["text"])],
                )
            )

        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=system or self.system_instruction,
        )

        response = self.client.models.generate_content(
            model=self.settings.model,
            contents=conversation,
            config=config,
        )

        text = response.text

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response. Try again later."
            )

        return text.strip()
