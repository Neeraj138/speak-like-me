from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable
from urllib.parse import urlparse

from .detectors import _RejectRedirects
from .language import contains_telugu_or_mixed


LANGUAGE_SYSTEM_PROMPT = """You are a strict language filter for an English-only writing dataset.
Classify each independent chat message. Exclude a message if any meaningful part is Telugu,
including Telugu written in Latin/English letters (romanized Telugu), Telugu script, or
Telugu-English code mixing. Examples to exclude include: 'em cestunnav', 'enti bro are you
coming', 'nuvvu ekkada unnava', and 'avuna'. Keep casual English, abbreviations, typos,
internet slang, names, and privacy placeholders such as <PERSON_1>. Never translate or
rewrite text. Return only the IDs that must be excluded."""


@dataclass(slots=True)
class LlamaCppLanguageFilter:
    """ID-only language classifier restricted to a local llama.cpp server."""

    model: str = "local-model"
    base_url: str = "http://127.0.0.1:8080"
    timeout_seconds: float = 180.0
    max_batch_items: int = 80
    max_batch_characters: int = 9_000
    parallel_requests: int = 1
    progress_callback: Callable[[int, int], None] | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("local model URL must use HTTP on localhost or a loopback address")
        if not self.model.strip():
            raise ValueError("local model name cannot be empty")
        if self.max_batch_items < 1 or self.max_batch_characters < 100:
            raise ValueError("language-classifier batch limits are too small")
        if self.parallel_requests < 1:
            raise ValueError("parallel_requests must be positive")

    def excluded_ids(self, items: Iterable[tuple[str, str]]) -> set[str]:
        source_items = [(str(item_id), text) for item_id, text in items if text.strip()]
        excluded: set[str] = set()
        batches = list(
            _batches(
                source_items,
                max_items=self.max_batch_items,
                max_characters=self.max_batch_characters,
            )
        )
        completed = 0
        with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
            futures = [executor.submit(self._classify_batch, batch) for batch in batches]
            for future in as_completed(futures):
                excluded.update(future.result())
                completed += 1
                if self.progress_callback is not None:
                    self.progress_callback(completed, len(batches))

        # A deterministic safety net prevents obvious romanized Telugu from surviving a
        # malformed or mistaken model response. The local model remains the primary pass.
        excluded.update(
            item_id for item_id, text in source_items if contains_telugu_or_mixed(text)
        )
        return excluded

    def _classify_batch(self, items: list[tuple[str, str]]) -> set[str]:
        valid_ids = {item_id for item_id, _ in items}
        user_payload = {
            "messages": [
                {"id": item_id, "text": text[:2_000]}
                for item_id, text in items
            ]
        }
        schema = {
            "name": "language_filter",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "exclude_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["exclude_ids"],
                "additionalProperties": False,
            },
        }
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                "temperature": 0,
                "max_tokens": min(512, max(96, len(items) * 6)),
                "response_format": {"type": "json_schema", "json_schema": schema},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirects(),
        )
        with opener.open(request, timeout=self.timeout_seconds) as response:
            outer = json.loads(response.read().decode("utf-8"))
        try:
            content = outer["choices"][0]["message"]["content"]
            result = json.loads(content)
            values = result["exclude_ids"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("local language classifier returned invalid JSON") from exc
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise RuntimeError("local language classifier returned an invalid ID list")
        return {value for value in values if value in valid_ids}


def _batches(
    items: list[tuple[str, str]], *, max_items: int, max_characters: int
) -> Iterable[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    characters = 0
    for item in items:
        item_characters = len(item[1][:2_000])
        if batch and (len(batch) >= max_items or characters + item_characters > max_characters):
            yield batch
            batch = []
            characters = 0
        batch.append(item)
        characters += item_characters
    if batch:
        yield batch
