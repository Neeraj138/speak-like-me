from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import RedactionConfig
from .detectors import ContextDetector
from .models import ParsedExport
from .parsers import filter_threads, parse_export
from .redaction import Redactor, residual_risks


def run_pipeline(
    *,
    source: str,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    config: RedactionConfig,
    selected_threads: set[str] | None = None,
    detectors: tuple[ContextDetector, ...] = (),
    strict: bool = True,
) -> dict[str, Any]:
    parsed = parse_export(source, input_path)
    source_thread_count = len({message.source_thread_id for message in parsed.messages})
    if source == "telegram" and source_thread_count > 1 and not selected_threads:
        raise ValueError(
            "a multi-chat Telegram export requires at least one --thread selection; "
            "run list-chats first"
        )
    messages = filter_threads(parsed.messages, selected_threads or set())
    if source == "linkedin" and not selected_threads:
        raise ValueError(
            "LinkedIn requires at least one --thread selection; run list-chats first"
        )
    if not messages:
        raise ValueError("the selected input contains no usable messages")

    _validate_distinct_paths(input_path, output_path, report_path)
    redactor = Redactor(config=config, detectors=detectors)
    redacted = redactor.redact_messages(messages)
    if strict and redactor.stats.review_reasons:
        reasons = ", ".join(sorted(redactor.stats.review_reasons))
        raise ValueError(
            f"strict privacy check found unresolved risk categories: {reasons}; "
            "no output was written"
        )

    payload = "".join(
        json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for item in redacted
    )
    report = _build_report(parsed, redactor, source, selected_threads, bool(detectors), strict)
    _atomic_private_write(output_path, payload)
    _atomic_private_write(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report


def audit_sanitized(
    path: Path, config: RedactionConfig | None = None
) -> dict[str, Any]:
    messages = 0
    reasons: dict[str, int] = {}
    invalid_lines = 0
    known_private_term_hits = 0
    private_terms: tuple[str, ...] = ()
    if config is not None:
        private_terms = tuple(
            term.casefold()
            for term in (
                *config.self_names,
                *(value for values in config.custom_entities.values() for value in values),
            )
            if term.casefold() not in {value.casefold() for value in config.allowed_terms}
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            text = record["text"]
            if not isinstance(text, str):
                raise TypeError
        except (json.JSONDecodeError, KeyError, TypeError):
            invalid_lines += 1
            continue
        messages += 1
        folded_text = text.casefold()
        if any(term in folded_text for term in private_terms):
            known_private_term_hits += 1
        for reason in residual_risks(text):
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "schema_version": 1,
        "messages_scanned": messages,
        "invalid_lines": invalid_lines,
        "review_reasons": reasons,
        "known_private_term_hits": known_private_term_hits,
        "passed": not reasons and invalid_lines == 0 and known_private_term_hits == 0,
        "manual_review_required": True,
    }


def _build_report(
    parsed: ParsedExport,
    redactor: Redactor,
    source: str,
    selected_threads: set[str] | None,
    contextual_detectors_enabled: bool,
    strict: bool,
) -> dict[str, Any]:
    stats = redactor.stats
    return {
        "schema_version": 1,
        "source": source,
        "input_file_count": len(parsed.input_files),
        "input_fingerprints": [_file_fingerprint(path) for path in parsed.input_files],
        "selected_thread_count": len(selected_threads or ()),
        "input_messages": stats.input_messages,
        "output_messages": stats.output_messages,
        "self_messages": stats.self_messages,
        "other_messages": stats.other_messages,
        "dropped_empty_messages": stats.dropped_empty_messages,
        "replacement_counts": dict(sorted(stats.replacements.items())),
        "review_reasons": dict(sorted(stats.review_reasons.items())),
        "strict_mode": strict,
        "contextual_detectors_enabled": contextual_detectors_enabled,
        "manual_review_required": True,
        "contains_raw_text": False,
    }


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_distinct_paths(input_path: Path, output_path: Path, report_path: Path) -> None:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    if output_path == report_path:
        raise ValueError("output and report paths must be different")
    if input_path.is_file() and input_path in {output_path, report_path}:
        raise ValueError("output files cannot overwrite the input export")
    if input_path.is_dir():
        for destination in (output_path, report_path):
            if input_path == destination or input_path in destination.parents:
                raise ValueError("outputs must be written outside the raw input directory")


def _atomic_private_write(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise
