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
from speak_like_me.language import contains_telugu_or_mixed
from speak_like_me.llamacpp import LlamaCppLanguageFilter
from speak_like_me.models import Message, RedactedMessage
from speak_like_me.parsers import parse_telegram, parse_whatsapp, parse_writings
from speak_like_me.redaction import Redactor, residual_risks


def main() -> int:
    args = _arguments()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    whatsapp_files = sorted(raw_root.glob("*.zip"))
    telegram_dirs = sorted(path for path in raw_root.glob("telegram*") if path.is_dir())
    writing_dirs = sorted(path for path in raw_root.glob("*journal*") if path.is_dir())
    if not whatsapp_files or len(telegram_dirs) != 1 or len(writing_dirs) != 1:
        raise ValueError("expected WhatsApp ZIPs, one Telegram directory, and one journal directory")

    whatsapp = parse_whatsapp(raw_root)
    whatsapp_self = _infer_common_whatsapp_self(whatsapp.messages)
    telegram = parse_telegram(telegram_dirs[0])
    telegram_self = _infer_telegram_self(telegram.messages, whatsapp_self)
    writings = parse_writings(writing_dirs[0])

    whatsapp_redactor = Redactor(RedactionConfig(self_names=(whatsapp_self,)))
    whatsapp_clean = whatsapp_redactor.redact_messages(whatsapp.messages)
    telegram_redactor = Redactor(RedactionConfig(self_names=(telegram_self,)))
    telegram_clean = telegram_redactor.redact_messages(telegram.messages)
    writing_redactor = Redactor(RedactionConfig(self_names=("__self__",)))
    writing_clean = _chunk_writings(writing_redactor.redact_messages(writings.messages))

    review_reasons = _merge_counts(
        whatsapp_redactor.stats.review_reasons,
        telegram_redactor.stats.review_reasons,
        writing_redactor.stats.review_reasons,
    )
    if review_reasons:
        categories = ", ".join(
            f"{name}={count}" for name, count in sorted(review_reasons.items())
        )
        raise ValueError(
            f"deterministic privacy audit failed ({categories}); no corpus was written"
        )

    all_messages = [*whatsapp_clean, *telegram_clean, *writing_clean]
    classifier = LlamaCppLanguageFilter(
        base_url=args.llamacpp_url,
        parallel_requests=4,
        progress_callback=lambda current, total: print(
            json.dumps({"language_batch": current, "language_batches": total}),
            flush=True,
        ),
    )
    candidates = [
        (str(index), item.text)
        for index, item in enumerate(all_messages)
    ]
    excluded_ids = classifier.excluded_ids(candidates)
    final_messages = [
        item for index, item in enumerate(all_messages) if str(index) not in excluded_ids
    ]
    heuristic_survivors = sum(contains_telugu_or_mixed(item.text) for item in final_messages)
    if heuristic_survivors:
        raise ValueError("language safety audit failed; no corpus was written")

    residual = Counter(
        reason for item in final_messages for reason in residual_risks(item.text)
    )
    if residual:
        raise ValueError("final privacy audit failed; no corpus was written")

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_path = output_dir / "sanitized_corpus.jsonl"
    report_path = output_dir / "processing_report.json"
    corpus = "".join(
        json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for item in final_messages
    )
    by_source_before = Counter(item.source for item in all_messages)
    by_source_after = Counter(item.source for item in final_messages)
    report = {
        "schema_version": 1,
        "input_files": {
            "whatsapp": len(whatsapp.input_files),
            "telegram": len(telegram.input_files),
            "writing": len(writings.input_files),
        },
        "messages_before_language_filter": dict(sorted(by_source_before.items())),
        "messages_after_language_filter": dict(sorted(by_source_after.items())),
        "messages_excluded_as_non_english_or_mixed": len(excluded_ids),
        "self_messages_after_filter": sum(item.author_role == "self" for item in final_messages),
        "other_messages_after_filter": sum(item.author_role == "other" for item in final_messages),
        "replacement_counts": _merge_counts(
            whatsapp_redactor.stats.replacements,
            telegram_redactor.stats.replacements,
            writing_redactor.stats.replacements,
        ),
        "deterministic_privacy_audit_passed": True,
        "romanized_telugu_safety_audit_passed": True,
        "language_classifier": "local llama.cpp; ID-only response; deterministic safety net",
        "contains_raw_text": False,
        "manual_review_required_before_training": True,
    }
    _private_write(output_path, corpus)
    _private_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", **report}, indent=2, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize the private personal-writing corpus")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llamacpp-url", default="http://127.0.0.1:8080")
    return parser.parse_args()


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
        (
            (len(target_tokens & _name_tokens(sender)), count, sender)
            for sender, count in counts.items()
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] == 0:
        raise ValueError("could not infer the Telegram self identity")
    if len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]:
        raise ValueError("Telegram self identity is ambiguous")
    return ranked[0][2]


def _name_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 1}


def _chunk_writings(messages: list[RedactedMessage], limit: int = 2_500) -> list[RedactedMessage]:
    result: list[RedactedMessage] = []
    order_by_thread: dict[str, int] = defaultdict(int)
    for message in messages:
        chunks: list[str] = []
        current = ""
        for paragraph in re.split(r"\n\s*\n", message.text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if current and len(current) + len(paragraph) + 2 > limit:
                chunks.append(current)
                current = ""
            while len(paragraph) > limit:
                split_at = paragraph.rfind(" ", 0, limit)
                split_at = split_at if split_at > limit // 2 else limit
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(paragraph[:split_at].strip())
                paragraph = paragraph[split_at:].strip()
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current:
            chunks.append(current)
        for chunk in chunks:
            order = order_by_thread[message.thread_id]
            order_by_thread[message.thread_id] += 1
            result.append(
                RedactedMessage(
                    source=message.source,
                    thread_id=message.thread_id,
                    order=order,
                    author_role=message.author_role,
                    speaker=message.speaker,
                    text=chunk,
                )
            )
    return result


def _merge_counts(*values: dict[str, int]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for value in values:
        result.update(value)
    return dict(sorted(result.items()))


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
