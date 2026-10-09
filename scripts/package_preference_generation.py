#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path


FILES = (
    Path("training/generate_preference_candidates.py"),
    Path("training/requirements.txt"),
    Path("data/v1/preference/prompts.jsonl"),
    Path("data/v1/preference/rubric.json"),
)
ADAPTER = Path("artifacts/casual-chat-qwen3-1.7b-lora/final_adapter")


def main() -> None:
    output = Path("data/v1/training/preference-generation-colab.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in FILES:
                archive.write(path, path.as_posix())
            for path in sorted(ADAPTER.iterdir()):
                if path.is_file():
                    archive.write(path, path.as_posix())
        temporary.replace(output)
        output.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
