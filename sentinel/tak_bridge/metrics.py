"""TAK bridge metrics."""

from __future__ import annotations


def compute_tak_metrics(messages: list[dict]) -> dict:
    n = len(messages)
    if n == 0:
        return {
            "message_count": 0,
            "sent_count": 0,
            "failed_count": 0,
            "categories": {},
            "types": {},
        }

    sent = sum(1 for m in messages if m.get("success"))
    failed = n - sent
    categories: dict[str, int] = {}
    types: dict[str, int] = {}

    for m in messages:
        cat = m.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        t = m.get("type", "unknown")
        types[t] = types.get(t, 0) + 1

    return {
        "message_count": n,
        "sent_count": sent,
        "failed_count": failed,
        "categories": categories,
        "types": types,
    }
