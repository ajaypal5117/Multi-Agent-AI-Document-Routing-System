"""Email agent: parse headers, then pull sender, urgency and the ask."""

import re
from email import message_from_string
from email.utils import parseaddr

from llm import extract_fields

WANTED = ["sender", "subject", "urgency", "request", "summary"]

URGENT_CUES = ["urgent", "asap", "immediately", "critical", "escalate", "deadline"]


def _body(message):
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode("utf-8", errors="replace")
        return ""
    payload = message.get_payload(decode=True)
    if payload is None:
        return str(message.get_payload())
    return payload.decode("utf-8", errors="replace")


def _rule_fields(headers, body):
    lowered = body.lower()
    urgency = "high" if any(cue in lowered for cue in URGENT_CUES) else "normal"
    request = None
    for line in body.splitlines():
        if re.search(r"\b(please|could you|request|kindly|need)\b", line, re.IGNORECASE):
            request = line.strip()
            break
    return {
        "sender": headers.get("sender"),
        "subject": headers.get("subject"),
        "urgency": urgency,
        "request": request,
        "summary": body.strip()[:200] or None,
    }


def process(path=None, raw=None):
    text = raw if raw is not None else open(path, encoding="utf-8", errors="replace").read()
    message = message_from_string(text)

    headers = {
        "sender": parseaddr(message.get("From", ""))[1] or message.get("From"),
        "subject": message.get("Subject"),
        "date": message.get("Date"),
    }
    body = _body(message) or text

    fields = extract_fields(f"{headers}\n\n{body[:3000]}", WANTED) or _rule_fields(headers, body)
    fields.setdefault("sender", headers["sender"])
    return {"agent": "email_agent", "headers": headers, "fields": fields}
