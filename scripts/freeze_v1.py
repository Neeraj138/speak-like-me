#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from speak_like_me.pipeline import _atomic_private_write


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def main() -> None:
    root = Path("data/v1/casual_chat")
    files = [root / "train.jsonl", root / "validation.jsonl"]
    manifest = {
        "dataset_version": "casual-chat-v1",
        "status": "frozen_for_baseline_training",
        "owner_review": "waived_by_owner; automated audits passed",
        "files": {
            path.name: {"examples": count_lines(path), "sha256": digest(path)} for path in files
        },
        "training_policy": {
            "loss": "assistant completion only",
            "validation_boundary": "whole conversation",
            "publish_dataset": False,
        },
    }
    _atomic_private_write(root / "frozen_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
