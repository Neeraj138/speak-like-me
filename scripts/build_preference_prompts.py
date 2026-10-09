#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from speak_like_me.dataset import load_jsonl
from speak_like_me.pipeline import _atomic_private_write
from speak_like_me.preference import select_preference_prompts, to_preference_record


def main() -> None:
    validation = load_jsonl(Path("data/v1/casual_chat/validation.jsonl"))
    selected = select_preference_prompts(validation)
    records = [to_preference_record(item, index) for index, item in enumerate(selected, 1)]
    output = Path("data/v1/preference/prompts.jsonl")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    _atomic_private_write(output, payload)
    print(json.dumps({"prompts": len(records), "source": "held-out validation conversation"}, indent=2))


if __name__ == "__main__":
    main()
