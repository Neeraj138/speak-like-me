from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RedactionConfig:
    self_names: tuple[str, ...]
    custom_entities: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allowed_terms: tuple[str, ...] = ()
    redact_urls: bool = True
    redact_handles: bool = True
    require_self_messages: bool = True

    @classmethod
    def from_file(cls, path: Path) -> "RedactionConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        self_names = tuple(_clean_terms(raw.get("self_names", [])))
        if not self_names:
            raise ValueError("config must contain at least one non-empty self_names value")

        custom_entities: dict[str, tuple[str, ...]] = {}
        for label, values in raw.get("custom_entities", {}).items():
            clean_label = _normalize_label(label)
            clean_values = tuple(_clean_terms(values))
            if clean_values:
                custom_entities[clean_label] = clean_values

        return cls(
            self_names=self_names,
            custom_entities=custom_entities,
            allowed_terms=tuple(_clean_terms(raw.get("allowed_terms", []))),
            redact_urls=bool(raw.get("redact_urls", True)),
            redact_handles=bool(raw.get("redact_handles", True)),
            require_self_messages=bool(raw.get("require_self_messages", True)),
        )


def _clean_terms(values: object) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("configuration term collections must be JSON arrays")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("configuration terms must be strings")
        value = value.strip()
        if value:
            result.append(value)
    return result


def _normalize_label(label: object) -> str:
    if not isinstance(label, str):
        raise ValueError("custom entity labels must be strings")
    normalized = "_".join(label.strip().upper().split())
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError(f"invalid custom entity label: {label!r}")
    return normalized

