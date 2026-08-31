"""PDF agent: extract text, then pull invoice-style fields."""

import re

from llm import extract_fields

WANTED = ["sender", "document_date", "invoice_number", "total_amount", "currency", "summary"]

AMOUNT = re.compile(r"(?:total|amount)\s*(?:due|payable)?\s*[:\-]?\s*([₹$€£]?\s?[\d,]+\.?\d{0,2})",
                    re.IGNORECASE)
INVOICE_NO = re.compile(r"invoice\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE)
DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")


def read_text(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _regex_fields(text):
    amount = AMOUNT.search(text)
    number = INVOICE_NO.search(text)
    date = DATE.search(text)
    return {
        "sender": None,
        "document_date": date.group(1) if date else None,
        "invoice_number": number.group(1) if number else None,
        "total_amount": amount.group(1).strip() if amount else None,
        "currency": None,
        "summary": text.strip()[:200] or None,
    }


def process(path):
    text = read_text(path)
    fields = extract_fields(text[:4000], WANTED) or _regex_fields(text)
    return {"agent": "pdf_agent", "text_chars": len(text), "fields": fields}
