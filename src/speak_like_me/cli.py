from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import RedactionConfig
from .detectors import OllamaDetector, PresidioDetector
from .parsers import parse_export, thread_summaries
from .pipeline import audit_sanitized, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slm-redact",
        description="Offline-first redaction for personal writing exports",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    redact = subparsers.add_parser("redact", help="parse and redact an export")
    redact.add_argument("--source", choices=["whatsapp", "telegram", "linkedin", "writing"], required=True)
    redact.add_argument("--input", type=Path, required=True)
    redact.add_argument("--config", type=Path, required=True)
    redact.add_argument("--output", type=Path, required=True)
    redact.add_argument("--report", type=Path, required=True)
    redact.add_argument(
        "--thread",
        action="append",
        default=[],
        help="conversation selection ID or private ID; repeat to select multiple",
    )
    redact.add_argument("--ollama-model", help="enable context detection with a local Ollama model")
    redact.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="loopback-only Ollama base URL",
    )
    redact.add_argument("--presidio", action="store_true", help="enable locally installed Presidio")
    redact.add_argument(
        "--allow-review-flags",
        action="store_true",
        help="write output even when residual deterministic checks require review",
    )

    list_chats = subparsers.add_parser(
        "list-chats", help="list conversations locally without showing message text"
    )
    list_chats.add_argument("--source", choices=["linkedin", "telegram"], required=True)
    list_chats.add_argument("--input", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="scan sanitized JSONL for residual deterministic risks")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--config", type=Path, help="also verify configured private terms are absent")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-chats":
            summaries = thread_summaries(parse_export(args.source, args.input))
            print(json.dumps(summaries, ensure_ascii=False, indent=2))
            return 0

        if args.command == "audit":
            config = RedactionConfig.from_file(args.config) if args.config else None
            report = audit_sanitized(args.input, config=config)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["passed"] else 2

        config = RedactionConfig.from_file(args.config)
        detectors = []
        if args.presidio:
            detectors.append(PresidioDetector())
        if args.ollama_model:
            detectors.append(OllamaDetector(model=args.ollama_model, base_url=args.ollama_url))
        report = run_pipeline(
            source=args.source,
            input_path=args.input,
            output_path=args.output,
            report_path=args.report,
            config=config,
            selected_threads=set(args.thread),
            detectors=tuple(detectors),
            strict=not args.allow_review_flags,
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "output_messages": report["output_messages"],
                    "manual_review_required": True,
                    "report": str(args.report),
                },
                indent=2,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
