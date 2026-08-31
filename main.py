"""Entry point: classify an input, route it to the right agent, persist the result.

    python main.py samples/invoice.pdf
    python main.py samples/order.json samples/complaint.eml --thread order-4471
    python main.py --history order-4471
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import memory
from agents import classifier, email_agent, json_agent, pdf_agent

AGENTS = {
    "pdf": pdf_agent,
    "json": json_agent,
    "email": email_agent,
}


def preview_text(path, fmt):
    """A short text sample the classifier can read the intent from."""
    if fmt == "pdf":
        try:
            return pdf_agent.read_text(path)[:3000]
        except Exception:
            return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")[:3000]


def route(path, thread_id, conn):
    fmt = classifier.detect_format(path=path)
    intent, confidence = classifier.detect_intent(preview_text(path, fmt))

    agent = AGENTS[fmt]
    result = agent.process(path) if fmt == "pdf" else agent.process(path=path)

    document_id = memory.log_document(conn, thread_id, path, fmt, intent, confidence)
    memory.log_fields(conn, document_id, result["agent"], result["fields"])

    return {
        "document_id": document_id,
        "source": str(path),
        "format": fmt,
        "intent": intent,
        "confidence": confidence,
        "agent": result["agent"],
        "fields": result["fields"],
    }


def main():
    parser = argparse.ArgumentParser(description="Route documents to format-specific agents.")
    parser.add_argument("inputs", nargs="*", help="Files to process")
    parser.add_argument("--thread", help="Group inputs under one conversation thread")
    parser.add_argument("--history", metavar="THREAD_ID", help="Print a stored thread and exit")
    args = parser.parse_args()

    conn = memory.connect()

    if args.history:
        print(json.dumps(memory.get_thread(conn, args.history), indent=2))
        return 0

    if not args.inputs:
        parser.error("give at least one file, or use --history")

    thread_id = args.thread or f"thread-{uuid.uuid4().hex[:8]}"
    print(f"thread: {thread_id}\n")

    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.exists():
            print(f"  skipped {path} (not found)")
            continue
        result = route(path, thread_id, conn)
        print(f"  {path.name}")
        print(f"    format {result['format']} | intent {result['intent']} "
              f"({result['confidence']}) -> {result['agent']}")
        for key, value in result["fields"].items():
            if value not in (None, ""):
                flat = " ".join(str(value).split())
                if len(flat) > 100:
                    flat = flat[:97] + "..."
                print(f"    {key}: {flat}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
