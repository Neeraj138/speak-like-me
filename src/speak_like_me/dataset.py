from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .language import contains_telugu_or_mixed


PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")
URL_ONLY_RE = re.compile(r"^(?:<URL>|https?://\S+)(?:\s+(?:<URL>|https?://\S+))*$", re.I)
MEDIA_RE = re.compile(
    r"(?:<media omitted>|image omitted|video omitted|audio omitted|sticker omitted|"
    r"gif omitted|document omitted|this message was deleted)",
    re.I,
)
WORD_RE = re.compile(r"[A-Za-z0-9']+")
BLOCKED_TERM_RE = re.compile(
    r"\b(?:nigg(?:a|as|er|ers)?|chink|fagg?(?:ot|ots)?|retard(?:ed|s)?)\b", re.I
)


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    text: str
    first_order: int
    last_order: int


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def clean_message(text: str) -> str:
    text = MEDIA_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_low_information(text: str) -> bool:
    if not text or URL_ONLY_RE.fullmatch(text):
        return True
    if PLACEHOLDER_RE.sub("", text).strip(" .,!?;:-_/\\") == "":
        return True
    return not WORD_RE.search(text) and len(text) < 8


def excessive_redaction(text: str) -> bool:
    placeholders = PLACEHOLDER_RE.findall(text)
    lexical_items = WORD_RE.findall(PLACEHOLDER_RE.sub("", text))
    return bool(placeholders) and len(placeholders) / (len(placeholders) + len(lexical_items)) > 0.35


def unsafe_text(text: str) -> bool:
    return bool(BLOCKED_TERM_RE.search(text))


def merge_turns(records: Iterable[dict[str, Any]]) -> list[Turn]:
    turns: list[Turn] = []
    for record in sorted(records, key=lambda row: int(row["order"])):
        role = "assistant" if record["author_role"] == "self" else "user"
        text = clean_message(str(record["text"]))
        if is_low_information(text):
            continue
        order = int(record["order"])
        if turns and turns[-1].role == role:
            previous = turns[-1]
            turns[-1] = Turn(role, f"{previous.text}\n{text}", previous.first_order, order)
        else:
            turns.append(Turn(role, text, order, order))
    return turns


def thread_language_rate(records: Iterable[dict[str, Any]]) -> tuple[int, int, float]:
    rows = list(records)
    hits = sum(contains_telugu_or_mixed(str(row["text"])) for row in rows)
    return hits, len(rows), hits / len(rows) if rows else 0.0


def build_examples(
    records: Iterable[dict[str, Any]],
    *,
    max_context_turns: int = 6,
    max_context_chars: int = 4000,
    max_target_chars: int = 1200,
    duplicate_completion_limit: int = 10,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = list(records)
    if not rows:
        return [], Counter()
    source = str(rows[0]["source"])
    thread_id = str(rows[0]["thread_id"])
    turns = merge_turns(rows)
    stats: Counter[str] = Counter(turns=len(turns))
    examples: list[dict[str, Any]] = []
    completion_counts: Counter[str] = Counter()
    safe_context: list[Turn] = []

    for turn in turns:
        bad_language = contains_telugu_or_mixed(turn.text)
        bad_content = unsafe_text(turn.text)
        bad_redaction = excessive_redaction(turn.text)
        if bad_language or bad_content or bad_redaction:
            reason = "language" if bad_language else "blocked_term" if bad_content else "redaction"
            stats[f"reset_{reason}"] += 1
            safe_context.clear()
            continue

        if turn.role != "assistant":
            safe_context.append(turn)
            safe_context = safe_context[-max_context_turns:]
            continue

        if len(turn.text) > max_target_chars:
            stats["excluded_long_target"] += 1
        elif safe_context and safe_context[-1].role == "user":
            normalized = re.sub(r"\W+", " ", turn.text.casefold()).strip()
            if completion_counts[normalized] >= duplicate_completion_limit:
                stats["excluded_duplicate_completion"] += 1
            else:
                context: list[Turn] = []
                chars = 0
                for prior in reversed(safe_context[-max_context_turns:]):
                    if context and chars + len(prior.text) > max_context_chars:
                        break
                    context.append(prior)
                    chars += len(prior.text)
                context.reverse()
                if context and context[-1].role == "user":
                    key = hashlib.sha256(
                        (thread_id + "\0" + str(turn.first_order) + "\0" + turn.text).encode()
                    ).hexdigest()[:16]
                    examples.append(
                        {
                            "example_id": key,
                            "prompt": [
                                {
                                    "role": "system",
                                    "content": "Reply naturally in casual English and match the conversation's tone and length.",
                                },
                                *({"role": item.role, "content": item.text} for item in context),
                            ],
                            "completion": [{"role": "assistant", "content": turn.text}],
                            "metadata": {
                                "source": source,
                                "thread_id": thread_id,
                                "target_order": turn.first_order,
                            },
                        }
                    )
                    completion_counts[normalized] += 1
                    stats["examples"] += 1
        else:
            stats["excluded_no_user_context"] += 1

        safe_context.append(turn)
        safe_context = safe_context[-max_context_turns:]
    return examples, stats


def group_by_thread(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["thread_id"])].append(record)
    return dict(groups)


def select_validation_thread(example_counts: dict[str, int], fraction: float = 0.15) -> str:
    total = sum(example_counts.values())
    if len(example_counts) < 2:
        raise ValueError("at least two usable conversations are required for a thread-level split")
    target = total * fraction
    return min(example_counts, key=lambda key: (abs(example_counts[key] - target), key))
