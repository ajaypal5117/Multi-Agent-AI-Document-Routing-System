"""Classifier agent.

Two decisions, made differently on purpose:

* **Format** (PDF / JSON / Email) is decided by rules — magic bytes, a JSON
  parse attempt, RFC-822 headers. Format is a structural fact, and rules are
  faster, free and more reliable than an LLM for it.
* **Intent** (invoice, complaint, RFQ, regulation, other) is decided by an LLM,
  because it depends on what the text means. If no API key is configured, a
  keyword fallback keeps the pipeline runnable offline.

The classifier only decides; `router.py` acts on the decision.
"""

import json
import re
from pathlib import Path

from llm import classify_intent

INTENTS = ["invoice", "complaint", "rfq", "regulation", "other"]

# Fallback keyword cues, used only when no LLM is available.
CUES = {
    "invoice": ["invoice", "amount due", "total due", "bill to", "payment terms", "gst"],
    "complaint": ["complaint", "dissatisfied", "unacceptable", "refund", "escalate", "faulty"],
    "rfq": ["request for quote", "rfq", "quotation", "pricing for", "tender"],
    "regulation": ["regulation", "compliance", "circular", "policy", "clause", "statute"],
}


def detect_format(path=None, raw=None):
    """Return 'pdf', 'json' or 'email' from the content itself, not the extension."""
    if path is not None:
        data = Path(path).read_bytes()
    else:
        data = raw if isinstance(raw, bytes) else str(raw).encode()

    if data[:5] == b"%PDF-":
        return "pdf"

    try:
        text = data.decode("utf-8", errors="replace").strip()
    except Exception:
        return "pdf"

    if text[:1] in "{[":
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError:
            pass

    # RFC-822 style headers at the top of the message.
    if re.search(r"^(From|To|Subject|Date)\s*:", text[:600], re.MULTILINE | re.IGNORECASE):
        return "email"

    return "email"  # plain text is treated as an email body


def _fallback_intent(text):
    lowered = text.lower()
    scores = {
        intent: sum(lowered.count(cue) for cue in cues)
        for intent, cues in CUES.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other", 0.3
    confidence = min(0.5 + 0.15 * scores[best], 0.9)
    return best, round(confidence, 2)


def detect_intent(text):
    """Return (intent, confidence). Falls back to keywords without an API key."""
    result = classify_intent(text[:3000], INTENTS)
    if result is None:
        return _fallback_intent(text)

    intent = result.get("intent", "other")
    if intent not in INTENTS:
        intent = "other"
    try:
        confidence = round(float(result.get("confidence", 0.5)), 2)
    except (TypeError, ValueError):
        confidence = 0.5
    return intent, confidence


def classify(path=None, raw=None, preview=""):
    fmt = detect_format(path=path, raw=raw)
    intent, confidence = detect_intent(preview)
    return {"format": fmt, "intent": intent, "confidence": confidence}
