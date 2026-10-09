from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from .models import DetectedEntity


class ContextDetector(Protocol):
    def detect(self, text: str) -> list[DetectedEntity]: ...


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise RuntimeError("local model server attempted an HTTP redirect")


@dataclass(slots=True)
class OllamaDetector:
    """Context-sensitive entity detector restricted to a local Ollama server."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("local model URL must use HTTP on localhost or a loopback address")
        if not self.model.strip():
            raise ValueError("local model name cannot be empty")

    def detect(self, text: str) -> list[DetectedEntity]:
        prompt = (
            "Find private or identifying entities in the text. Do not rewrite the text. "
            "Return JSON only as {\"entities\":[{\"text\":\"exact substring\","
            "\"label\":\"PERSON|ORGANIZATION|LOCATION|ADDRESS|PROJECT|IDENTIFIER\"}]}. "
            "Include personal names, employers, clients, private project names, exact places, "
            "street addresses, and context-specific identifiers. Ignore generic technologies "
            "and ordinary non-identifying nouns. Text follows:\n\n" + text
        )
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/generate",
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
        inner = json.loads(outer.get("response", "{}"))
        return _validate_entities(inner.get("entities", []), text)


@dataclass(slots=True)
class PresidioDetector:
    """Optional local Presidio adapter. No text leaves the running process."""

    language: str = "en"
    model_name: str = "en_core_web_lg"
    _analyzer: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError as exc:
            raise RuntimeError(
                "Presidio is not installed. Use Python 3.12 or 3.13 and install the presidio extra."
            ) from exc
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": self.language, "model_name": self.model_name}],
            }
        )
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=[self.language],
        )

    def detect(self, text: str) -> list[DetectedEntity]:
        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            entities=[
                "PERSON",
                "LOCATION",
                "ORGANIZATION",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "IP_ADDRESS",
                "CREDIT_CARD",
            ],
        )
        entities = [
            DetectedEntity(text=text[result.start : result.end], label=result.entity_type)
            for result in results
            if result.end > result.start
        ]
        return entities


def _validate_entities(values: object, source_text: str) -> list[DetectedEntity]:
    if not isinstance(values, list):
        return []
    allowed_labels = {
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "ADDRESS",
        "PROJECT",
        "IDENTIFIER",
    }
    result: list[DetectedEntity] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        text = value.get("text")
        label = value.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            continue
        label = label.upper().strip()
        if label not in allowed_labels or not text.strip() or text not in source_text:
            continue
        key = (text, label)
        if key not in seen:
            seen.add(key)
            result.append(DetectedEntity(text=text, label=label))
    return result
