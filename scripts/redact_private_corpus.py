#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from speak_like_me.config import RedactionConfig
from speak_like_me.detectors import PresidioDetector
from speak_like_me.models import Message, ParsedExport, RedactedMessage
from speak_like_me.parsers import parse_telegram, parse_whatsapp, parse_writings
from speak_like_me.redaction import Redactor, residual_risks


def main() -> int:
    args = _arguments()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    whatsapp_files = sorted(raw_root.glob("*.txt"))
    telegram_dirs = sorted(path for path in raw_root.glob("telegram*") if path.is_dir())
    writing_dirs = sorted(path for path in raw_root.glob("*journal*") if path.is_dir())
    if not whatsapp_files or len(telegram_dirs) != 1 or len(writing_dirs) != 1:
        raise ValueError("expected extracted WhatsApp text files and one Telegram and journal directory")

    whatsapp = _combine(parse_whatsapp(path) for path in whatsapp_files)
    telegram = parse_telegram(telegram_dirs[0])
    writings = parse_writings(writing_dirs[0])
    whatsapp_self = _infer_common_whatsapp_self(whatsapp.messages)
    telegram_self = _infer_telegram_self(telegram.messages, whatsapp_self)

    # Presidio's email recognizer uses the public-suffix list. Point it at a local
    # project cache so the real redaction pass does not attempt any network access.
    suffix_cache = Path.cwd() / ".cache" / "tldextract"
    suffix_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["TLDEXTRACT_CACHE"] = str(suffix_cache)
    presidio = PresidioDetector(model_name="en_core_web_lg")
    whatsapp_clean, whatsapp_redactor = _redact(
        "whatsapp", whatsapp.messages, (whatsapp_self,), presidio
    )
    telegram_clean, telegram_redactor = _redact(
        "telegram", telegram.messages, (telegram_self,), presidio
    )
    writing_clean, writing_redactor = _redact(
        "writing", writings.messages, ("__self__",), presidio
    )

    all_clean = [*whatsapp_clean, *telegram_clean, *writing_clean]
    risks = Counter(reason for item in all_clean for reason in residual_risks(item.text))
    if risks:
        categories = ", ".join(f"{name}={count}" for name, count in sorted(risks.items()))
        raise ValueError(f"residual privacy audit failed ({categories}); no files were written")

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_jsonl(output_dir / "whatsapp.redacted.jsonl", whatsapp_clean)
    _write_jsonl(output_dir / "telegram.redacted.jsonl", telegram_clean)
    _write_jsonl(output_dir / "writings.redacted.jsonl", writing_clean)
    _write_transcripts(review_dir, all_clean)

    replacements = Counter()
    for redactor in (whatsapp_redactor, telegram_redactor, writing_redactor):
        replacements.update(redactor.stats.replacements)
    report = {
        "schema_version": 1,
        "status": "redaction_completed",
        "presidio_model": "en_core_web_lg",
        "presidio_runs_locally": True,
        "input_files": {
            "whatsapp": len(whatsapp.input_files),
            "telegram": len(telegram.input_files),
            "writing": len(writings.input_files),
        },
        "output_messages": dict(
            sorted(Counter(item.source for item in all_clean).items())
        ),
        "replacement_counts": dict(sorted(replacements.items())),
        "residual_privacy_audit_passed": True,
        "contains_raw_text": False,
        "language_filtering_performed": False,
        "user_review_required_before_assistant_reads_text": True,
    }
    _private_write(
        output_dir / "redaction_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redact the private corpus without language filtering")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _combine(exports) -> ParsedExport:
    messages: list[Message] = []
    files: list[Path] = []
    for parsed in exports:
        messages.extend(parsed.messages)
        files.extend(parsed.input_files)
    return ParsedExport(tuple(messages), tuple(files))


def _progress(source: str):
    def report(current: int, total: int) -> None:
        if current == total or current % 500 == 0:
            print(json.dumps({"source": source, "redacted": current, "total": total}), flush=True)
    return report


def _redact(
    source: str,
    messages: tuple[Message, ...],
    self_names: tuple[str, ...],
    detector: PresidioDetector,
) -> tuple[list[RedactedMessage], Redactor]:
    redactor = Redactor(
        RedactionConfig(self_names=self_names),
        detectors=(detector,),
        progress_callback=_progress(source),
    )
    return redactor.redact_messages(messages), redactor


def _infer_common_whatsapp_self(messages: tuple[Message, ...]) -> str:
    speakers_by_thread: dict[str, set[str]] = defaultdict(set)
    display_names: dict[str, str] = {}
    for message in messages:
        folded = message.sender.casefold().strip()
        speakers_by_thread[message.source_thread_id].add(folded)
        display_names.setdefault(folded, message.sender)
    common = set.intersection(*speakers_by_thread.values()) if speakers_by_thread else set()
    if len(common) != 1:
        raise ValueError("could not uniquely infer the common WhatsApp self identity")
    return display_names[common.pop()]


def _infer_telegram_self(messages: tuple[Message, ...], whatsapp_self: str) -> str:
    counts = Counter(message.sender for message in messages)
    target_tokens = _name_tokens(whatsapp_self)
    ranked = sorted(
        ((len(target_tokens & _name_tokens(sender)), count, sender) for sender, count in counts.items()),
        reverse=True,
    )
    if not ranked or ranked[0][0] == 0:
        raise ValueError("could not infer the Telegram self identity")
    if len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]:
        raise ValueError("Telegram self identity is ambiguous")
    return ranked[0][2]


def _name_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 1}


def _write_jsonl(path: Path, messages: list[RedactedMessage]) -> None:
    content = "".join(
        json.dumps(message.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for message in messages
    )
    _private_write(path, content)


def _write_transcripts(directory: Path, messages: list[RedactedMessage]) -> None:
    grouped: dict[tuple[str, str], list[RedactedMessage]] = defaultdict(list)
    for message in messages:
        grouped[(message.source, message.thread_id)].append(message)
    counters: Counter[str] = Counter()
    for (source, _thread_id), items in grouped.items():
        counters[source] += 1
        lines = []
        for item in sorted(items, key=lambda value: value.order):
            text = item.text.replace("\r\n", "\n").replace("\r", "\n")
            lines.append(f"[{item.speaker}] {text}")
        _private_write(
            directory / f"{source}_{counters[source]:04d}.txt",
            "\n\n".join(lines) + "\n",
        )


def _private_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        path.chmod(0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
