from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    source: str
    source_thread_id: str
    order: int
    sender: str
    text: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedExport:
    messages: tuple[Message, ...]
    input_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RedactedMessage:
    source: str
    thread_id: str
    order: int
    author_role: str
    speaker: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "thread_id": self.thread_id,
            "order": self.order,
            "author_role": self.author_role,
            "speaker": self.speaker,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class DetectedEntity:
    text: str
    label: str


@dataclass(slots=True)
class RedactionStats:
    input_messages: int = 0
    output_messages: int = 0
    self_messages: int = 0
    other_messages: int = 0
    dropped_empty_messages: int = 0
    replacements: dict[str, int] = field(default_factory=dict)
    review_reasons: dict[str, int] = field(default_factory=dict)

    def add_replacement(self, placeholder: str, count: int = 1) -> None:
        self.replacements[placeholder] = self.replacements.get(placeholder, 0) + count

    def add_review_reason(self, reason: str, count: int = 1) -> None:
        self.review_reasons[reason] = self.review_reasons.get(reason, 0) + count

