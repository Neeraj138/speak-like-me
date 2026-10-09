from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

from .models import Message, ParsedExport


_WHATSAPP_BRACKETED = re.compile(
    r"^\[(?P<date>[^,\]]+),\s*(?P<time>[^\]]+)\]\s*(?P<body>.*)$"
)
_WHATSAPP_DASHED = re.compile(
    r"^(?P<date>\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][.]?[Mm][.]?)?)\s+-\s+"
    r"(?P<body>.*)$"
)


def parse_export(source: str, input_path: Path) -> ParsedExport:
    source = source.lower()
    if source == "whatsapp":
        return parse_whatsapp(input_path)
    if source == "telegram":
        return parse_telegram(input_path)
    if source == "linkedin":
        return parse_linkedin(input_path)
    if source == "writing":
        return parse_writings(input_path)
    raise ValueError(f"unsupported source: {source}")


def parse_whatsapp(input_path: Path) -> ParsedExport:
    files = _collect_files(input_path, {".txt", ".zip"})
    messages: list[Message] = []
    for path in files:
        content = _read_whatsapp_content(path)
        source_thread_id = _private_thread_id("whatsapp", content)
        current: dict[str, str] | None = None
        order = 0

        for raw_line in content.splitlines():
            line = raw_line.lstrip("\u200e\u200f")
            match = _WHATSAPP_BRACKETED.match(line) or _WHATSAPP_DASHED.match(line)
            if match:
                if current is not None:
                    messages.append(_whatsapp_message(source_thread_id, order, current))
                    order += 1
                body = match.group("body")
                if ": " not in body:
                    current = None  # system or service event
                    continue
                sender, text = body.split(": ", 1)
                current = {
                    "sender": sender.strip(),
                    "text": text,
                    "timestamp": f"{match.group('date')} {match.group('time')}",
                }
            elif current is not None:
                current["text"] += "\n" + raw_line

        if current is not None:
            messages.append(_whatsapp_message(source_thread_id, order, current))

    return ParsedExport(tuple(messages), tuple(files))


def _read_whatsapp_content(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8-sig", errors="replace")
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() == ".txt"
        ]
        if len(candidates) != 1:
            raise ValueError(
                "WhatsApp ZIP must contain exactly one text export and no extraction is performed"
            )
        return archive.read(candidates[0]).decode("utf-8-sig", errors="replace")


def _whatsapp_message(thread_id: str, order: int, current: dict[str, str]) -> Message:
    return Message(
        source="whatsapp",
        source_thread_id=thread_id,
        order=order,
        sender=current["sender"],
        text=current["text"],
        timestamp=current["timestamp"],
    )


def parse_telegram(input_path: Path) -> ParsedExport:
    files = _collect_files(input_path, {".json", ".html"})
    if files and all(path.suffix.lower() == ".html" for path in files):
        return _parse_telegram_html(files)
    messages: list[Message] = []
    for path in (path for path in files if path.suffix.lower() == ".json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        chats = _telegram_chats(raw)
        for chat_number, chat in enumerate(chats):
            raw_thread_key = str(chat.get("id") or chat.get("name") or chat_number)
            thread_id = _private_thread_id("telegram", raw_thread_key)
            order = 0
            for item in chat.get("messages", []):
                if item.get("type", "message") != "message":
                    continue
                text = _telegram_text(item.get("text", ""))
                sender = str(item.get("from") or item.get("actor") or "unknown").strip()
                if not text.strip() or not sender:
                    continue
                messages.append(
                    Message(
                        source="telegram",
                        source_thread_id=thread_id,
                        order=order,
                        sender=sender,
                        text=text,
                        timestamp=item.get("date"),
                        metadata={"selection_id": raw_thread_key},
                    )
                )
                order += 1
    return ParsedExport(tuple(messages), tuple(files))


def _parse_telegram_html(files: list[Path]) -> ParsedExport:
    ordered_files = sorted(files, key=_telegram_html_sort_key)
    thread_seed = "|".join(str(path.parent.resolve()) for path in ordered_files)
    thread_id = _private_thread_id("telegram", thread_seed)
    messages: list[Message] = []
    previous_sender = ""
    order = 0
    for path in ordered_files:
        parser = _TelegramHTMLParser(previous_sender=previous_sender)
        parser.feed(path.read_text(encoding="utf-8-sig", errors="replace"))
        parser.close()
        previous_sender = parser.previous_sender
        for sender, text, timestamp in parser.messages:
            if not sender or not text.strip():
                continue
            messages.append(
                Message(
                    source="telegram",
                    source_thread_id=thread_id,
                    order=order,
                    sender=sender,
                    text=text,
                    timestamp=timestamp,
                    metadata={"selection_id": thread_id},
                )
            )
            order += 1
    return ParsedExport(tuple(messages), tuple(ordered_files))


def _telegram_html_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 1, path.name)


class _TelegramHTMLParser(HTMLParser):
    def __init__(self, previous_sender: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[tuple[str, str, str | None]] = []
        self.previous_sender = previous_sender
        self._message_depth: int | None = None
        self._div_depth = 0
        self._field: str | None = None
        self._field_depth: int | None = None
        self._sender_parts: list[str] = []
        self._text_parts: list[str] = []
        self._timestamp: str | None = None
        self._is_forwarded = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag == "div":
            self._div_depth += 1
            classes = set((attrs_map.get("class") or "").split())
            if self._message_depth is None and {"message", "default"}.issubset(classes):
                self._message_depth = self._div_depth
                self._sender_parts = []
                self._text_parts = []
                self._timestamp = None
                self._is_forwarded = False
            elif self._message_depth is not None:
                if "forwarded" in classes and "body" in classes:
                    self._is_forwarded = True
                elif "from_name" in classes and not self._is_forwarded:
                    self._field = "sender"
                    self._field_depth = self._div_depth
                elif "text" in classes and not self._is_forwarded:
                    self._field = "text"
                    self._field_depth = self._div_depth
                elif "date" in classes and "details" in classes:
                    self._timestamp = attrs_map.get("title")
        elif tag == "br" and self._field == "text":
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag != "div":
            return
        if self._field_depth == self._div_depth:
            self._field = None
            self._field_depth = None
        if self._message_depth == self._div_depth:
            sender = "".join(self._sender_parts).strip() or self.previous_sender
            text = "".join(self._text_parts).strip()
            if sender:
                self.previous_sender = sender
            if sender and text and not self._is_forwarded:
                self.messages.append((sender, text, self._timestamp))
            self._message_depth = None
            self._sender_parts = []
            self._text_parts = []
            self._timestamp = None
            self._is_forwarded = False
            self._field = None
            self._field_depth = None
        self._div_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._field == "sender":
            self._sender_parts.append(data)
        elif self._field == "text":
            self._text_parts.append(data)


def _telegram_chats(raw: object) -> list[dict]:
    if not isinstance(raw, dict):
        raise ValueError("Telegram export must contain a JSON object")
    if isinstance(raw.get("messages"), list):
        return [raw]
    chats = raw.get("chats")
    if isinstance(chats, dict) and isinstance(chats.get("list"), list):
        return [item for item in chats["list"] if isinstance(item, dict)]
    raise ValueError("unrecognized Telegram JSON export structure")


def _telegram_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def parse_linkedin(input_path: Path) -> ParsedExport:
    files = _collect_files(input_path, {".csv"})
    messages: list[Message] = []
    order_by_thread: dict[str, int] = defaultdict(int)

    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            columns = {_column_key(name): name for name in reader.fieldnames}
            conversation_column = _find_column(
                columns, "conversationid", "conversation", "threadid", "thread"
            )
            sender_column = _find_column(columns, "from", "sender", "sendername")
            content_column = _find_column(columns, "content", "message", "body")
            date_column = _find_column(columns, "date", "timestamp", required=False)

            for row_number, row in enumerate(reader, start=1):
                raw_thread = (row.get(conversation_column) or "").strip()
                sender = (row.get(sender_column) or "").strip()
                text = row.get(content_column) or ""
                if not raw_thread or not sender or not text.strip():
                    continue
                thread_id = _private_thread_id("linkedin", raw_thread)
                order = order_by_thread[thread_id]
                order_by_thread[thread_id] += 1
                messages.append(
                    Message(
                        source="linkedin",
                        source_thread_id=thread_id,
                        order=order,
                        sender=sender,
                        text=text,
                        timestamp=(row.get(date_column) if date_column else None),
                        metadata={"selection_id": raw_thread, "row": row_number},
                    )
                )
    return ParsedExport(tuple(messages), tuple(files))


def thread_summaries(parsed: ParsedExport) -> list[dict[str, object]]:
    grouped: dict[str, list[Message]] = defaultdict(list)
    for message in parsed.messages:
        grouped[message.source_thread_id].append(message)

    result = []
    for private_id, items in grouped.items():
        participants = sorted({item.sender for item in items}, key=str.casefold)
        selection_id = str(items[0].metadata.get("selection_id", ""))
        dates = [item.timestamp for item in items if item.timestamp]
        result.append(
            {
                "selection_id": selection_id,
                "private_id": private_id,
                "participants": participants,
                "message_count": len(items),
                "first_date": min(dates) if dates else None,
                "last_date": max(dates) if dates else None,
            }
        )
    return sorted(result, key=lambda item: str(item["selection_id"]))


def parse_writings(input_path: Path) -> ParsedExport:
    files = _collect_files(input_path, {".txt", ".md", ".jsonl", ".docx"})
    messages: list[Message] = []
    for path in files:
        if path.suffix.lower() == ".jsonl":
            records = _read_writing_jsonl(path)
        elif path.suffix.lower() == ".docx":
            records = [_read_docx_text(path)]
        else:
            records = [path.read_text(encoding="utf-8-sig", errors="replace")]
        for order, text in enumerate(records):
            if text.strip():
                messages.append(
                    Message(
                        source="writing",
                        source_thread_id=_private_thread_id("writing", path.read_bytes()),
                        order=order,
                        sender="__self__",
                        text=text,
                    )
                )
    return ParsedExport(tuple(messages), tuple(files))


def _read_docx_text(path: Path) -> str:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(namespace + "p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == namespace + "t" and node.text:
                parts.append(node.text)
            elif node.tag == namespace + "tab":
                parts.append("\t")
            elif node.tag in {namespace + "br", namespace + "cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _read_writing_jsonl(path: Path) -> list[str]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, dict) and isinstance(value.get("text"), str):
            result.append(value["text"])
        else:
            raise ValueError(f"writing JSONL line {line_number} must be a string or contain text")
    return result


def _collect_files(input_path: Path, suffixes: set[str]) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
        )
    else:
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    files = [path for path in files if path.suffix.lower() in suffixes]
    if not files:
        expected = ", ".join(sorted(suffixes))
        raise ValueError(f"no supported input files found; expected: {expected}")
    return files


def _private_thread_id(source: str, value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    digest = hashlib.sha256(source.encode("ascii") + b"\0" + payload).hexdigest()[:16]
    return f"{source}_{digest}"


def _column_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _find_column(
    columns: dict[str, str], *candidates: str, required: bool = True
) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    if required:
        raise ValueError(f"missing required CSV column; tried: {', '.join(candidates)}")
    return None


def filter_threads(messages: Iterable[Message], selected_ids: set[str]) -> list[Message]:
    if not selected_ids:
        return list(messages)
    return [
        message
        for message in messages
        if message.source_thread_id in selected_ids
        or str(message.metadata.get("selection_id", "")) in selected_ids
    ]
