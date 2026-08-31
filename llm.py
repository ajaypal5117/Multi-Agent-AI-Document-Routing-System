"""Thin OpenAI wrapper.

Every function here returns None when no API key is set or the call fails, so
the agents can fall back to rules and the pipeline stays runnable offline. That
also keeps the tests free of network calls.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _client():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=key)
    except Exception:
        return None


def _json_call(system, user):
    client = _client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None


def classify_intent(text, intents):
    return _json_call(
        "You classify business documents. Reply with JSON only: "
        f'{{"intent": one of {intents}, "confidence": a number between 0 and 1}}. '
        "Base the intent on what the document is for, not its file format.",
        text,
    )


def extract_fields(text, wanted):
    """Ask for a specific set of fields. Missing values must come back as null."""
    return _json_call(
        "Extract fields from the document. Reply with JSON only, using exactly these keys: "
        f"{wanted}. Use null for anything not present. Never invent values.",
        text,
    )
