from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Callable

from .config import RedactionConfig
from .detectors import ContextDetector
from .models import DetectedEntity, Message, RedactedMessage, RedactionStats


PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    re.DOTALL,
)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")
BARE_URL_RE = re.compile(
    r"(?i)(?<![@\w])(?:[a-z0-9-]+\.)+(?:com|org|net|io|in|co|ai|dev|app|me|xyz)"
    r"(?:/[^\s<>]*)?"
)
IP_RE = re.compile(r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)")
HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.-]{1,31}\b")
UPI_RE = re.compile(r"(?i)(?<![\w.-])[a-z0-9._-]{2,256}@[a-z]{2,64}(?![\w.-])")
PAN_RE = re.compile(r"(?i)(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])")
IFSC_RE = re.compile(r"(?i)(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])")
AADHAAR_RE = re.compile(r"(?<!\d)(?<!\d[ -])(?:\d{4}[ -]?){2}\d{4}(?![ -]?\d)")
PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{5,}\d)(?!\w)")
LONG_ID_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
PASSWORD_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|api[_ -]?key|access[_ -]?token|secret)"
    r"(\s*(?::|=|\bis\b)\s*)([^\s,;]+)"
)
OTP_RE = re.compile(
    r"(?i)\b(otp|one[ -]?time password|verification code|pin)\b"
    r"(\s*(?::|=|\bis\b)\s*|\s+)(\d{4,10})(?!\d)"
)
REDACTED_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|api[_ -]?key|access[_ -]?token|secret)"
    r"\s*(?::|=|\bis\b)\s*<SECRET>"
)
REDACTED_OTP_RE = re.compile(
    r"(?i)\b(?:otp|one[ -]?time password|verification code|pin)\b"
    r"(?:\s*(?::|=|\bis\b)\s*|\s+)<OTP>"
)
BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")
KNOWN_SECRET_RE = re.compile(
    r"\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b"
)
PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*(?:_\d+)?>")


class Redactor:
    def __init__(
        self,
        config: RedactionConfig,
        detectors: Iterable[ContextDetector] = (),
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.detectors = tuple(detectors)
        self.progress_callback = progress_callback
        self.stats = RedactionStats()
        self._entity_placeholders: dict[tuple[str, str], str] = {}
        self._label_counts: dict[str, int] = defaultdict(int)
        self._allowed = {term.casefold() for term in config.allowed_terms}

    def redact_messages(self, messages: Iterable[Message]) -> list[RedactedMessage]:
        source_messages = list(messages)
        self.stats.input_messages = len(source_messages)
        speakers = self._speaker_maps(source_messages)
        thread_aliases = self._thread_aliases(source_messages)
        output: list[RedactedMessage] = []

        for message_number, message in enumerate(source_messages, start=1):
            text = message.text
            if not text.strip():
                self.stats.dropped_empty_messages += 1
                continue
            speaker = speakers[message.source_thread_id][message.sender.casefold()]
            is_self = speaker == "<SELF>"
            if is_self:
                self.stats.self_messages += 1
            else:
                self.stats.other_messages += 1

            text = self._redact_text(text, speakers[message.source_thread_id])
            review_reasons = residual_risks(text)
            for reason in review_reasons:
                self.stats.add_review_reason(reason)

            output.append(
                RedactedMessage(
                    source=message.source,
                    thread_id=thread_aliases[message.source_thread_id],
                    order=message.order,
                    author_role="self" if is_self else "other",
                    speaker=speaker,
                    text=text,
                )
            )
            if self.progress_callback is not None:
                self.progress_callback(message_number, len(source_messages))

        self.stats.output_messages = len(output)
        if self.config.require_self_messages and output and self.stats.self_messages == 0:
            raise ValueError(
                "no self-authored messages were detected; check self_names in the private config"
            )
        return output

    def _speaker_maps(self, messages: list[Message]) -> dict[str, dict[str, str]]:
        self_names = {value.casefold() for value in self.config.self_names}
        self_names.add("__self__")
        result: dict[str, dict[str, str]] = defaultdict(dict)
        counters: dict[str, int] = defaultdict(int)
        for message in messages:
            key = message.sender.casefold()
            if key in result[message.source_thread_id]:
                continue
            if key in self_names:
                result[message.source_thread_id][key] = "<SELF>"
            else:
                counters[message.source_thread_id] += 1
                result[message.source_thread_id][key] = (
                    f"<PERSON_{counters[message.source_thread_id]}>"
                )
        return result

    @staticmethod
    def _thread_aliases(messages: list[Message]) -> dict[str, str]:
        result: dict[str, str] = {}
        count_by_source: dict[str, int] = defaultdict(int)
        for message in messages:
            if message.source_thread_id not in result:
                count_by_source[message.source] += 1
                result[message.source_thread_id] = (
                    f"{message.source}_thread_{count_by_source[message.source]:04d}"
                )
        return result

    def _redact_text(self, text: str, speaker_map: dict[str, str]) -> str:
        # Presidio runs first so its NER model sees the original local text. The
        # deterministic passes below then catch structured values and known names.
        for detector in self.detectors:
            entities = detector.detect(text)
            text = self._apply_entities(text, entities)

        text = self._replace_regex(text, PRIVATE_KEY_RE, "<PRIVATE_KEY>")
        text = self._replace_regex(text, KNOWN_SECRET_RE, "<SECRET>")
        text = self._replace_regex(text, BEARER_RE, "<SECRET>")
        text = PASSWORD_ASSIGNMENT_RE.sub(self._password_replacement, text)
        text = OTP_RE.sub(self._otp_replacement, text)

        for label, values in self.config.custom_entities.items():
            for value in sorted(values, key=len, reverse=True):
                if self._literal_pattern(value).search(text):
                    text = self._replace_literal(text, value, self._placeholder(label, value))

        for raw_name, placeholder in sorted(speaker_map.items(), key=lambda item: len(item[0]), reverse=True):
            if raw_name not in self._allowed:
                text = self._replace_literal(text, raw_name, placeholder)

        text = self._replace_regex(text, EMAIL_RE, "<EMAIL>")
        text = self._replace_regex(text, UPI_RE, "<UPI_ID>")
        text = self._replace_regex(text, IP_RE, "<IP_ADDRESS>")
        text = self._replace_regex(text, PAN_RE, "<PAN>")
        text = self._replace_regex(text, IFSC_RE, "<IFSC>")
        text = self._replace_regex(text, AADHAAR_RE, "<AADHAAR>")
        if self.config.redact_urls:
            text = self._replace_regex(text, URL_RE, "<URL>")
            text = self._replace_regex(text, BARE_URL_RE, "<URL>")
        if self.config.redact_handles:
            text = self._replace_regex(text, HANDLE_RE, "<HANDLE>")
        text = CARD_CANDIDATE_RE.sub(self._card_replacement, text)
        text = PHONE_CANDIDATE_RE.sub(self._phone_replacement, text)
        text = self._replace_regex(text, LONG_ID_RE, "<ID_NUMBER>")
        return text

    def _apply_entities(self, text: str, entities: Iterable[DetectedEntity]) -> str:
        for entity in sorted(entities, key=lambda item: len(item.text), reverse=True):
            if entity.text.casefold() in self._allowed or PLACEHOLDER_RE.fullmatch(entity.text):
                continue
            if self._literal_pattern(entity.text).search(text):
                placeholder = self._placeholder(entity.label, entity.text)
                text = self._replace_literal(text, entity.text, placeholder)
        return text

    def _placeholder(self, label: str, value: str) -> str:
        label = _safe_label(label)
        key = (label, value.casefold())
        if key not in self._entity_placeholders:
            self._label_counts[label] += 1
            self._entity_placeholders[key] = f"<{label}_{self._label_counts[label]}>"
        return self._entity_placeholders[key]

    def _replace_literal(self, text: str, value: str, placeholder: str) -> str:
        if not value or value.casefold() in self._allowed:
            return text
        return self._replace_regex(text, self._literal_pattern(value), placeholder)

    @staticmethod
    def _literal_pattern(value: str) -> re.Pattern[str]:
        prefix = r"(?<!\w)" if value[0].isalnum() else ""
        suffix = r"(?!\w)" if value[-1].isalnum() else ""
        return re.compile(prefix + re.escape(value) + suffix, re.IGNORECASE)

    def _replace_regex(self, text: str, pattern: re.Pattern[str], placeholder: str) -> str:
        text, count = pattern.subn(placeholder, text)
        if count:
            self.stats.add_replacement(placeholder, count)
        return text

    def _password_replacement(self, match: re.Match[str]) -> str:
        self.stats.add_replacement("<SECRET>")
        return f"{match.group(1)}{match.group(2)}<SECRET>"

    def _otp_replacement(self, match: re.Match[str]) -> str:
        self.stats.add_replacement("<OTP>")
        return f"{match.group(1)}{match.group(2)}<OTP>"

    def _phone_replacement(self, match: re.Match[str]) -> str:
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        # The candidate pattern starts at six digits. Redact the full candidate range so
        # strict mode cannot leave a short local number or partial identifier behind.
        if len(digits) >= 6:
            self.stats.add_replacement("<PHONE>")
            return "<PHONE>"
        return value

    def _card_replacement(self, match: re.Match[str]) -> str:
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        # Privacy review favors recall: redact all card-shaped candidates even when a
        # typo or test number fails the checksum.
        if 13 <= len(digits) <= 19:
            self.stats.add_replacement("<CARD_NUMBER>")
            return "<CARD_NUMBER>"
        return value


def residual_risks(text: str) -> set[str]:
    audit_text = REDACTED_CREDENTIAL_RE.sub("", text)
    audit_text = REDACTED_OTP_RE.sub("", audit_text)
    # A word sentinel prevents removing a placeholder from joining or bridging text
    # on each side into a synthetic URL or number sequence.
    without_placeholders = PLACEHOLDER_RE.sub(" REDACTED ", audit_text)
    reasons = set()
    if PRIVATE_KEY_RE.search(without_placeholders):
        reasons.add("private_key")
    if KNOWN_SECRET_RE.search(without_placeholders) or BEARER_RE.search(without_placeholders):
        reasons.add("known_secret")
    if EMAIL_RE.search(without_placeholders):
        reasons.add("email")
    if UPI_RE.search(without_placeholders):
        reasons.add("upi_id")
    if URL_RE.search(without_placeholders) or BARE_URL_RE.search(without_placeholders):
        reasons.add("url")
    if PAN_RE.search(without_placeholders):
        reasons.add("pan")
    if IFSC_RE.search(without_placeholders):
        reasons.add("ifsc")
    if AADHAAR_RE.search(without_placeholders):
        reasons.add("aadhaar")
    if PASSWORD_ASSIGNMENT_RE.search(without_placeholders) or OTP_RE.search(without_placeholders):
        reasons.add("credential")
    if PHONE_CANDIDATE_RE.search(without_placeholders):
        reasons.add("number_sequence")
    return reasons


def _safe_label(label: str) -> str:
    label = "_".join(label.upper().strip().split())
    aliases = {
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "IP_ADDRESS": "IP_ADDRESS",
        "CREDIT_CARD": "CARD_NUMBER",
        "ORGANIZATION": "ORGANIZATION",
    }
    label = aliases.get(label, label)
    cleaned = "".join(character for character in label if character.isalnum() or character == "_")
    return cleaned or "PRIVATE_ENTITY"
