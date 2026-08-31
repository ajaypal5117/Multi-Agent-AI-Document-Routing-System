"""Shared memory: one SQLite database every agent reads and writes.

Two tables. `documents` holds one row per input processed (source, format,
intent, timestamp). `fields` holds the values agents extracted from it, keyed
back to the document. That gives cross-agent traceability: given any extracted
field you can find which file it came from and which agent pulled it.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    source      TEXT NOT NULL,
    format      TEXT NOT NULL,
    intent      TEXT NOT NULL,
    confidence  REAL,
    received_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fields (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    agent       TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT,
    extracted_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fields_doc ON fields(document_id);
CREATE INDEX IF NOT EXISTS idx_docs_thread ON documents(thread_id);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_document(conn, thread_id, source, fmt, intent, confidence):
    cur = conn.execute(
        "INSERT INTO documents (thread_id, source, format, intent, confidence, received_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (thread_id, str(source), fmt, intent, confidence, _now()),
    )
    conn.commit()
    return cur.lastrowid


def log_fields(conn, document_id, agent, fields):
    """Persist a dict of extracted fields. Non-scalar values are stored as JSON."""
    rows = [
        (
            document_id,
            agent,
            key,
            value if isinstance(value, str) or value is None else json.dumps(value),
            _now(),
        )
        for key, value in fields.items()
    ]
    conn.executemany(
        "INSERT INTO fields (document_id, agent, key, value, extracted_at)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def get_thread(conn, thread_id):
    """Every document in a thread, each with its extracted fields."""
    documents = conn.execute(
        "SELECT * FROM documents WHERE thread_id = ? ORDER BY id", (thread_id,)
    ).fetchall()

    result = []
    for doc in documents:
        fields = conn.execute(
            "SELECT agent, key, value FROM fields WHERE document_id = ?", (doc["id"],)
        ).fetchall()
        record = dict(doc)
        record["fields"] = {row["key"]: row["value"] for row in fields}
        record["agents"] = sorted({row["agent"] for row in fields})
        result.append(record)
    return result


def recent(conn, limit=20):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM documents ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()]
