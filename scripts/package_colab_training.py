#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path


FILES = (
    Path("training/train_qlora.py"),
    Path("training/requirements.txt"),
    Path("training/README.md"),
    Path("data/v1/casual_chat/train.jsonl"),
    Path("data/v1/casual_chat/validation.jsonl"),
    Path("data/v1/casual_chat/frozen_manifest.json"),
)


def main() -> None:
    output = Path("data/v1/training/casual-chat-v1-colab.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in FILES:
                if not path.is_file():
                    raise FileNotFoundError(path)
                archive.write(path, path.as_posix())
        temporary.replace(output)
        output.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
