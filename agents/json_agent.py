"""JSON agent: validate a payload against a target schema and flag what is missing.

This agent deliberately does not call an LLM. A JSON payload is already
structured, so the useful work is validation and reformatting, which rules do
exactly and cheaply.
"""

import json

# Fields the downstream system expects on every payload.
REQUIRED = ["id", "type", "timestamp", "payload"]


def _flatten(obj, prefix=""):
    flat = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        flat[prefix.rstrip(".")] = f"[{len(obj)} items]"
    else:
        flat[prefix.rstrip(".")] = obj
    return flat


def process(path=None, raw=None):
    text = raw if raw is not None else open(path, encoding="utf-8").read()
    data = json.loads(text)

    present = data.keys() if isinstance(data, dict) else {}
    missing = [key for key in REQUIRED if key not in present]

    flat = _flatten(data)
    anomalies = []
    if missing:
        anomalies.append(f"missing required keys: {', '.join(missing)}")
    if isinstance(data, dict) and not data:
        anomalies.append("empty object")

    fields = {
        "record_id": str(data.get("id")) if isinstance(data, dict) and "id" in data else None,
        "record_type": str(data.get("type")) if isinstance(data, dict) and "type" in data else None,
        "key_count": str(len(flat)),
        "missing_required": ", ".join(missing) if missing else None,
        "valid": "no" if anomalies else "yes",
    }
    return {"agent": "json_agent", "anomalies": anomalies, "fields": fields}
