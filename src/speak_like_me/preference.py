from __future__ import annotations

import hashlib
from typing import Any


def _rank(identifier: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{identifier}".encode()).hexdigest()


def select_preference_prompts(
    examples: list[dict[str, Any]], count: int = 100, salt: str = "casual-v1-preference"
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for example in examples:
        length = len(example["completion"][0]["content"])
        bucket = "short" if length <= 20 else "medium" if length <= 160 else "long"
        buckets[bucket].append(example)
    quotas = {"short": round(count * 0.3), "medium": round(count * 0.5)}
    quotas["long"] = count - quotas["short"] - quotas["medium"]
    chosen: dict[str, dict[str, Any]] = {}
    for bucket, rows in buckets.items():
        for row in sorted(rows, key=lambda item: _rank(item["example_id"], salt + bucket))[: quotas[bucket]]:
            chosen[row["example_id"]] = row
    if len(chosen) < count:
        remaining = [item for item in examples if item["example_id"] not in chosen]
        for row in sorted(remaining, key=lambda item: _rank(item["example_id"], salt))[: count - len(chosen)]:
            chosen[row["example_id"]] = row
    return sorted(chosen.values(), key=lambda item: _rank(item["example_id"], salt + "display"))


def to_preference_record(example: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "preference_id": f"casual_pref_{index:03d}",
        "prompt": example["prompt"],
        "reference_completion": example["completion"][0]["content"],
        "metadata": {
            "held_out_example_id": example["example_id"],
            "source": example["metadata"]["source"],
        },
    }
