# Multi-Agent AI Document Routing System

Accepts a PDF, JSON payload or email, works out what it is, hands it to the
agent that knows how to read that format, and writes everything to a shared
SQLite memory so the whole thread can be traced afterwards.

```
input ──▶ classifier ──┬──▶ pdf_agent    ──┐
                       ├──▶ json_agent   ──┼──▶ shared memory (SQLite)
                       └──▶ email_agent  ──┘
```

## How the classification splits

Format and intent are decided differently, on purpose:

- **Format** (pdf / json / email) is decided by rules — PDF magic bytes, a JSON
  parse attempt, RFC-822 headers. It's a structural fact, so rules are faster,
  free, and more reliable than a model. An extension is never trusted; a `.txt`
  containing `%PDF-` is routed as a PDF.
- **Intent** (invoice / complaint / rfq / regulation / other) is decided by an
  LLM, because it depends on what the text means. Without an API key a keyword
  fallback takes over so the pipeline still runs end to end.

## Shared memory

Two tables. `documents` holds one row per input (thread, source, format, intent,
confidence, timestamp). `fields` holds each extracted value keyed back to its
document and the agent that produced it. So from any field you can trace which
file it came from and which agent read it, and `--history` replays an entire
thread across agents.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: add OPENAI_API_KEY for LLM extraction

python main.py samples/invoice.pdf samples/order.json samples/complaint.eml --thread order-4471
python main.py --history order-4471
```

Output:

```
  invoice.pdf
    format pdf | intent invoice (0.9) -> pdf_agent
    invoice_number: INV-2025-0917
    total_amount: 184,500.00

  complaint.eml
    format email | intent complaint (0.65) -> email_agent
    sender: priya.nair@northwind-retail.com
    urgency: high
    request: Please arrange a replacement shipment immediately and confirm by end of day.
```

## Agents

| Agent | Reads | Produces |
|---|---|---|
| `pdf_agent` | text via pypdf | sender, date, invoice number, total, summary |
| `json_agent` | the payload itself | record id/type, key count, missing required keys, valid flag |
| `email_agent` | RFC-822 headers + body | sender, subject, urgency, the actual request |

`json_agent` uses no LLM at all — a JSON payload is already structured, so the
useful work is validation, which rules do exactly and for free.

## Tests

```bash
pytest -s
```

`test_format_accuracy` measures format classification against a labelled set
that includes the awkward cases (a JSON array, an email with no `From:` header,
text that looks like JSON but doesn't parse) and prints the score rather than
asserting a number from nowhere. Intent accuracy depends on the LLM and is not
asserted offline.

## Layout

```
├── main.py              routing + CLI
├── memory.py            SQLite shared memory
├── llm.py               OpenAI wrapper, returns None when unavailable
├── agents/
│   ├── classifier.py    format rules + intent classification
│   ├── pdf_agent.py
│   ├── json_agent.py
│   └── email_agent.py
├── test_router.py
└── samples/             one input of each format, plus one that fails validation
```

## Limits

- One document per file; no batch or queue intake.
- The offline intent fallback is keyword-based and noticeably weaker than the
  LLM path — structured payloads with no prose usually land on `other`.
- Scanned PDFs need OCR before they'll yield text.
