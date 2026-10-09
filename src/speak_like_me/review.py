from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import PLACEHOLDER_RE, load_jsonl
from .pipeline import _atomic_private_write


def _rank(example: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{example['example_id']}".encode()).hexdigest()


def review_focus(example: dict[str, Any]) -> list[str]:
    target = str(example["completion"][0]["content"])
    whole = " ".join(item["content"] for item in example["prompt"]) + " " + target
    focus: list[str] = []
    if PLACEHOLDER_RE.search(whole):
        focus.append("redaction_placeholder")
    if len(target) <= 12:
        focus.append("very_short_reply")
    if len(target) >= 300:
        focus.append("long_reply")
    return focus or ["general_quality"]


def select_review_sample(
    examples: list[dict[str, Any]], count: int, *, seed: str
) -> list[dict[str, Any]]:
    if count >= len(examples):
        selected = list(examples)
    else:
        chosen: dict[str, dict[str, Any]] = {}

        def take(candidates: list[dict[str, Any]], amount: int) -> None:
            for example in candidates:
                if len(chosen) >= count or amount <= 0:
                    break
                key = str(example["example_id"])
                if key not in chosen:
                    chosen[key] = example
                    amount -= 1

        placeholder = [
            item
            for item in examples
            if "redaction_placeholder" in review_focus(item)
        ]
        shortest = sorted(examples, key=lambda item: len(item["completion"][0]["content"]))
        longest = list(reversed(shortest))
        take(sorted(placeholder, key=lambda item: _rank(item, seed + ":placeholder")), max(1, count // 5))
        take(shortest, max(1, count // 6))
        take(longest, max(1, count // 6))
        take(sorted(examples, key=lambda item: _rank(item, seed + ":random")), count)
        selected = list(chosen.values())

    return sorted(selected, key=lambda item: _rank(item, seed + ":display"))


def prepare_review_records(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], *, train_count: int = 120,
    validation_count: int = 30,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split, examples, count in (
        ("train", train, train_count),
        ("validation", validation, validation_count),
    ):
        for example in select_review_sample(examples, count, seed=f"v1:{split}"):
            records.append(
                {
                    "example_id": example["example_id"],
                    "split": split,
                    "source": example["metadata"]["source"],
                    "focus": review_focus(example),
                    "prompt": example["prompt"],
                    "completion": example["completion"][0]["content"],
                }
            )
    return records


def build_review_pack(dataset_dir: Path, output_path: Path) -> dict[str, Any]:
    train = load_jsonl(dataset_dir / "train.jsonl")
    validation = load_jsonl(dataset_dir / "validation.jsonl")
    records = prepare_review_records(train, validation)
    template = (Path(__file__).with_name("review_template.html")).read_text(encoding="utf-8")
    embedded = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__REVIEW_DATA__", embedded)
    if "__REVIEW_DATA__" in html:
        raise RuntimeError("review data placeholder was not replaced")
    _atomic_private_write(output_path, html)
    return {
        "sample_size": len(records),
        "train_samples": sum(row["split"] == "train" for row in records),
        "validation_samples": sum(row["split"] == "validation" for row in records),
        "placeholder_samples": sum("redaction_placeholder" in row["focus"] for row in records),
        "short_samples": sum("very_short_reply" in row["focus"] for row in records),
        "long_samples": sum("long_reply" in row["focus"] for row in records),
    }
