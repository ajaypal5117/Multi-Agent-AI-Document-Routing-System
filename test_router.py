"""Tests. No network and no API key needed — the LLM layer falls back to rules.

`test_format_accuracy` is the one that matters: it measures format
classification against a labelled set rather than asserting a number.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import memory
from agents import classifier, email_agent, json_agent

# (raw input, expected format). Deliberately includes awkward cases: JSON with a
# leading array, an email with no headers, and JSON-looking text that isn't JSON.
LABELLED = [
    (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\ntrailer", "pdf"),
    ('{"id": 1, "type": "order"}', "json"),
    ('[{"a": 1}, {"b": 2}]', "json"),
    ('{"nested": {"deep": [1, 2, 3]}}', "json"),
    ("From: a@b.com\nSubject: hello\n\nBody text here.", "email"),
    ("To: ops@x.com\nFrom: y@z.com\nSubject: RFQ\n\nPlease quote.", "email"),
    ("Subject: no from header\n\nStill an email.", "email"),
    ("Just some plain text with no headers at all.", "email"),
    ('{"broken": ', "email"),          # looks like JSON, does not parse
    ("Date: Mon, 14 Oct 2025\n\nDated note.", "email"),
]


def test_format_accuracy():
    correct = sum(
        classifier.detect_format(raw=raw) == expected for raw, expected in LABELLED
    )
    accuracy = correct / len(LABELLED)
    print(f"\nformat accuracy: {accuracy:.0%} ({correct}/{len(LABELLED)})")
    assert accuracy >= 0.95


def test_pdf_detected_by_magic_bytes_not_extension():
    assert classifier.detect_format(raw=b"%PDF-1.4 rest") == "pdf"


def test_malformed_json_is_not_classified_as_json():
    assert classifier.detect_format(raw='{"a": 1') != "json"


def test_intent_fallback_finds_complaint():
    intent, confidence = classifier.detect_intent(
        "This is a complaint. The unit is faulty and I want a refund."
    )
    assert intent == "complaint"
    assert 0 < confidence <= 1


def test_intent_fallback_returns_other_when_no_cues():
    intent, confidence = classifier.detect_intent("The weather today is mild.")
    assert intent == "other"


def test_json_agent_flags_missing_required_keys():
    result = json_agent.process(raw='{"type": "po", "payload": {}}')
    assert result["fields"]["valid"] == "no"
    assert "id" in result["fields"]["missing_required"]


def test_json_agent_accepts_complete_payload():
    raw = json.dumps({"id": "1", "type": "po", "timestamp": "now", "payload": {"x": 1}})
    assert json_agent.process(raw=raw)["fields"]["valid"] == "yes"


def test_email_agent_parses_headers_and_urgency():
    raw = "From: p@q.com\nSubject: Urgent issue\n\nPlease escalate this immediately."
    fields = email_agent.process(raw=raw)["fields"]
    assert fields["sender"] == "p@q.com"
    assert fields["urgency"] == "high"


def test_memory_round_trip(tmp_path):
    conn = memory.connect(tmp_path / "test.db")
    doc_id = memory.log_document(conn, "t-1", "a.eml", "email", "complaint", 0.8)
    memory.log_fields(conn, doc_id, "email_agent", {"sender": "p@q.com", "urgency": "high"})

    thread = memory.get_thread(conn, "t-1")
    assert len(thread) == 1
    assert thread[0]["fields"]["sender"] == "p@q.com"
    assert thread[0]["agents"] == ["email_agent"]


def test_memory_groups_multiple_documents_in_one_thread(tmp_path):
    conn = memory.connect(tmp_path / "test.db")
    for source, fmt in [("a.pdf", "pdf"), ("b.json", "json")]:
        doc_id = memory.log_document(conn, "t-2", source, fmt, "invoice", 0.9)
        memory.log_fields(conn, doc_id, f"{fmt}_agent", {"k": "v"})
    assert len(memory.get_thread(conn, "t-2")) == 2
