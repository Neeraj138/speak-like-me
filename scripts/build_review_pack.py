#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from speak_like_me.review import build_review_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the private offline v1 review pack")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/v1/casual_chat"))
    parser.add_argument("--output", type=Path, default=Path("data/v1/review/casual_chat_review.html"))
    args = parser.parse_args()
    print(json.dumps(build_review_pack(args.dataset_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
