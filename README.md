# speak-like-me privacy pipeline

This project converts selected personal chat and writing exports into a sanitized,
source-neutral JSONL format. It is deliberately a redaction stage, not yet a training
dataset generator.

The safe operating boundary is:

```text
private raw exports (outside this repository)
        -> local parsing and redaction
        -> sanitized JSONL + content-free report
        -> manual review
        -> later dataset preparation
```

The core pipeline has no third-party Python dependencies. It supports:

- WhatsApp individual-chat text exports;
- Telegram Desktop JSON exports, including account exports containing several chats;
- LinkedIn Messages CSV exports, with mandatory explicit conversation selection;
- plain-text, Markdown and `{"text": "..."}` JSONL writings;
- deterministic masking of email addresses, phone numbers, URLs, handles, IP addresses,
  card numbers, credentials, tokens and private keys;
- stable placeholders for participants and configured names, organizations, projects and
  places;
- optional local-only context detection through Ollama or Presidio;
- strict residual-risk checks, private file permissions and reports that contain counts
  and hashes but no message text.

## Keep raw data outside the repository

Use a private directory such as:

```text
~/speak-like-me-private/
  raw/
    whatsapp/
    telegram/
    linkedin/
    writings/
  selected/
  sanitized/
  reports/
  redaction.private.json
```

The repository ignores `data/`, `private/`, and private configuration files, but keeping
raw exports outside the repository provides a stronger boundary. Do not email raw exports
or place them in a synchronized folder merely to run this tool.

## Run without installing

From this repository:

```bash
PYTHONPATH=src python -m speak_like_me --help
```

To install an isolated command locally:

```bash
uv venv
uv pip install -e .
```

## Create the private configuration

Copy `examples/synthetic/config.json` to the private directory and replace the fake values:

```json
{
  "self_names": ["Your exact sender name in the exports"],
  "custom_entities": {
    "person": ["private names mentioned in messages"],
    "company": ["employers and clients"],
    "project": ["private project names"],
    "location": ["identifying places"]
  },
  "allowed_terms": ["Python"],
  "redact_urls": true,
  "redact_handles": true,
  "require_self_messages": true
}
```

This configuration is itself private because it contains the terms being removed. Keep it
outside the repository. `self_names` must match the sender name used for your own messages.
If it is wrong, the command stops rather than silently producing a dataset with no known
self-authored messages.

## WhatsApp

Export each chosen chat without media and place its `.txt` file in the private raw folder.
Process a single export with:

```bash
PYTHONPATH=src python -m speak_like_me redact \
  --source whatsapp \
  --input ~/speak-like-me-private/raw/whatsapp/chat-001.txt \
  --config ~/speak-like-me-private/redaction.private.json \
  --output ~/speak-like-me-private/sanitized/chat-001.jsonl \
  --report ~/speak-like-me-private/reports/chat-001.json
```

The parser supports common Android and iPhone timestamp formats and preserves multiline
messages. WhatsApp service events are omitted.

## Telegram

Use Telegram Desktop's JSON export format and omit media. A single-chat export or an
account export containing multiple chats is supported:

```bash
PYTHONPATH=src python -m speak_like_me redact \
  --source telegram \
  --input ~/speak-like-me-private/raw/telegram/result.json \
  --config ~/speak-like-me-private/redaction.private.json \
  --output ~/speak-like-me-private/sanitized/telegram.jsonl \
  --report ~/speak-like-me-private/reports/telegram.json
```

For an account-wide Telegram export, list its chats locally and then select the required
IDs. Message bodies are not printed:

```bash
PYTHONPATH=src python -m speak_like_me list-chats \
  --source telegram \
  --input ~/speak-like-me-private/raw/telegram/result.json

PYTHONPATH=src python -m speak_like_me redact \
  --source telegram \
  --input ~/speak-like-me-private/raw/telegram/result.json \
  --thread 'chosen-chat-id' \
  --config ~/speak-like-me-private/redaction.private.json \
  --output ~/speak-like-me-private/sanitized/telegram.jsonl \
  --report ~/speak-like-me-private/reports/telegram.json
```

The command refuses to process a multi-chat Telegram export without a selection. A
single-chat export does not require `--thread`.

## LinkedIn

LinkedIn supplies one Messages CSV containing several conversations. First list the chats
locally; message bodies are never printed:

```bash
PYTHONPATH=src python -m speak_like_me list-chats \
  --source linkedin \
  --input ~/speak-like-me-private/raw/linkedin/Messages.csv
```

Select only the required conversation IDs:

```bash
PYTHONPATH=src python -m speak_like_me redact \
  --source linkedin \
  --input ~/speak-like-me-private/raw/linkedin/Messages.csv \
  --thread 'chosen-conversation-id' \
  --config ~/speak-like-me-private/redaction.private.json \
  --output ~/speak-like-me-private/sanitized/linkedin.jsonl \
  --report ~/speak-like-me-private/reports/linkedin.json
```

The command refuses to process LinkedIn data without at least one `--thread`. Repeat the
option to select several conversations.

## Other writings

Plain text, Markdown, and JSONL files are accepted:

```bash
PYTHONPATH=src python -m speak_like_me redact \
  --source writing \
  --input ~/speak-like-me-private/raw/writings/ \
  --config ~/speak-like-me-private/redaction.private.json \
  --output ~/speak-like-me-private/sanitized/writings.jsonl \
  --report ~/speak-like-me-private/reports/writings.json
```

The tool does not rewrite, correct or summarize your writing. It only masks detected text.

## Optional local model check

Ollama integration is disabled by default. When enabled, the program accepts only plain
HTTP endpoints on `localhost`, `127.0.0.1`, or `::1`; proxies and redirects are disabled.
The model is asked to identify exact substrings and cannot rewrite the message:

```bash
PYTHONPATH=src python -m speak_like_me redact \
  --source whatsapp \
  --input ~/speak-like-me-private/raw/whatsapp/chat-001.txt \
  --config ~/speak-like-me-private/redaction.private.json \
  --output ~/speak-like-me-private/sanitized/chat-001.jsonl \
  --report ~/speak-like-me-private/reports/chat-001.json \
  --ollama-model YOUR_LOCAL_MODEL
```

Verify that the model is already present locally before disconnecting the machine from the
network. Model-assisted detection remains a review aid and is not a privacy guarantee.

Presidio can also be enabled with `--presidio`. Its published Python support currently ends
at Python 3.13, so use a Python 3.12 or 3.13 environment for that optional integration:

```bash
uv venv --python 3.13
uv pip install -e '.[presidio]'
```

Presidio also requires a compatible local NLP model. Installation may download packages or
models; perform installation before switching to an offline run.

## Audit and manual review

Run the deterministic audit after every redaction:

```bash
PYTHONPATH=src python -m speak_like_me audit \
  --input ~/speak-like-me-private/sanitized/chat-001.jsonl \
  --config ~/speak-like-me-private/redaction.private.json
```

The default redaction command is strict: if a known deterministic pattern remains, it writes
no output. `--allow-review-flags` exists for debugging difficult exports, but such output is
not ready to share.

Even a passing audit requires manual review. Inspect a representative sample and search the
sanitized files for names, organizations, project names, usernames, exact places and rare
personal stories that the automated layers may not understand.

## Output schema

Each sanitized JSONL record has this structure:

```json
{
  "author_role": "self",
  "order": 1,
  "source": "whatsapp",
  "speaker": "<SELF>",
  "text": "yeah, ill send it after the <COMPANY_1> call",
  "thread_id": "whatsapp_thread_0001"
}
```

Exact timestamps, original thread names, sender names, input paths and raw message text are
not included in the output report. The `order` field retains conversation sequence without
retaining the original time.

## Test with synthetic data

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Everything under `examples/synthetic/` is fabricated and safe for development tests.
