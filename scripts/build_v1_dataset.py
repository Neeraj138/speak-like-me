#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from speak_like_me.dataset import (
    build_examples,
    group_by_thread,
    load_jsonl,
    select_validation_thread,
    thread_language_rate,
)
from speak_like_me.pipeline import _atomic_private_write


IDEAS = [
    ("travel", "When a carefully planned trip changes on arrival", "Adaptability matters more than a perfect itinerary."),
    ("travel", "What a student budget teaches about priorities", "Constraints can make decisions clearer and experiences more memorable."),
    ("travel", "Missing a destination without losing the trip", "A changed plan can still produce the strongest story."),
    ("travel", "The first encounter with snow", "Novel experiences restore curiosity that routine often suppresses."),
    ("travel", "A night journey seen through a bus window", "Periods of uncertainty can still contain moments worth noticing."),
    ("travel", "Famous places versus personal expectations", "Reputation shapes expectations, while direct experience creates the real opinion."),
    ("travel", "Famous food that did not match the hype", "Independent judgment grows when experience challenges a popular claim."),
    ("travel", "Choosing between activities when time is limited", "Opportunity cost becomes real when every option is attractive."),
    ("travel", "Learning from the destination that was skipped", "Unfinished plans can reveal what to prepare differently next time."),
    ("travel", "How unfamiliar public transport changes observation", "New systems make everyday design choices visible."),
    ("teamwork", "Friends changing plans when someone falls ill", "A good group measures success by how it treats people under strain."),
    ("teamwork", "Dividing travel logistics among friends", "Informal trips still teach ownership, coordination, and trust."),
    ("teamwork", "Keeping track of shared expenses", "Small acts of accounting prevent confusion and protect relationships."),
    ("communication", "Navigating a place despite a language barrier", "Clear intent, patience, and context can carry a conversation."),
    ("communication", "Asking strangers for help while travelling", "Specific questions make it easier for people to help."),
    ("resilience", "Continuing through fear or physical discomfort", "Confidence often follows action rather than preceding it."),
    ("resilience", "Recovering when several small plans fail", "Adaptability is a sequence of small decisions, not one dramatic pivot."),
    ("learning", "Why direct experience can teach more than a lecture", "A concept becomes durable when it is connected to something lived."),
    ("learning", "Architecture that triggered engineering curiosity", "Wonder can be the first step toward technical questions."),
    ("learning", "Re-reading old writing and noticing a changed perspective", "Growth becomes visible when earlier assumptions are preserved."),
    ("civic", "Public behaviour at monuments and shared spaces", "Civic responsibility appears in ordinary choices around common property."),
    ("civic", "Tourism and responsibility to a place", "Enjoying a destination includes reducing the burden placed on it."),
    ("reflection", "Turning a personal experience into a professional lesson", "The useful lesson should emerge from a specific event rather than a slogan."),
    ("reflection", "Planning versus adaptability", "Planning creates direction; adjustment keeps the plan useful when reality changes."),
]


def idea_records() -> list[dict[str, Any]]:
    return [
        {
            "brief_id": f"linkedin_idea_{index:03d}",
            "source_collection": "authorized_journals_and_reflections",
            "category": category,
            "topic": topic,
            "possible_takeaway": takeaway,
            "recommended_shape": "specific scene -> honest observation -> practical takeaway",
            "status": "needs_human_review_and_post_draft",
            "training_ready": False,
        }
        for index, (category, topic, takeaway) in enumerate(IDEAS, start=1)
    ]


def jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def cap_duplicate_completions(
    examples: list[dict[str, Any]], limit: int = 10
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    dropped = 0
    for example in examples:
        text = str(example["completion"][0]["content"])
        normalized = re.sub(r"\W+", " ", text.casefold()).strip()
        if counts[normalized] >= limit:
            dropped += 1
            continue
        counts[normalized] += 1
        kept.append(example)
    return kept, dropped


def build(redacted_dir: Path, output_dir: Path) -> dict[str, Any]:
    all_rows = load_jsonl(redacted_dir / "whatsapp.redacted.jsonl")
    all_rows.extend(load_jsonl(redacted_dir / "telegram.redacted.jsonl"))
    threads = group_by_thread(all_rows)

    exclusion_manifest: list[dict[str, Any]] = []
    usable: dict[str, list[dict[str, Any]]] = {}
    language_audit: dict[str, dict[str, Any]] = {}
    for thread_id, rows in sorted(threads.items()):
        hits, messages, rate = thread_language_rate(rows)
        language_audit[thread_id] = {
            "messages": messages,
            "language_signal_messages": hits,
            "language_signal_rate": round(rate, 6),
        }
        if rate >= 0.02:
            exclusion_manifest.append(
                {
                    "thread_id": thread_id,
                    "messages": messages,
                    "reason": "mixed_romanized_telugu_and_english_thread",
                }
            )
        else:
            usable[thread_id] = rows

    per_thread: dict[str, list[dict[str, Any]]] = {}
    filter_stats: Counter[str] = Counter()
    for thread_id, rows in usable.items():
        examples, stats = build_examples(rows)
        if examples:
            per_thread[thread_id] = examples
        filter_stats.update(stats)

    validation_thread = select_validation_thread(
        {thread_id: len(examples) for thread_id, examples in per_thread.items()}
    )
    train = [
        example
        for thread_id, examples in sorted(per_thread.items())
        if thread_id != validation_thread
        for example in examples
    ]
    validation = per_thread[validation_thread]
    train, train_global_duplicates = cap_duplicate_completions(train)
    validation, validation_global_duplicates = cap_duplicate_completions(validation)
    filter_stats["excluded_global_duplicate_completion"] += (
        train_global_duplicates + validation_global_duplicates
    )

    casual_dir = output_dir / "casual_chat"
    linkedin_dir = output_dir / "linkedin"
    _atomic_private_write(casual_dir / "train.jsonl", jsonl(train))
    _atomic_private_write(casual_dir / "validation.jsonl", jsonl(validation))
    _atomic_private_write(
        casual_dir / "excluded_threads.json",
        json.dumps(exclusion_manifest, indent=2, sort_keys=True) + "\n",
    )

    report = {
        "schema_version": 1,
        "dataset_version": "v1",
        "purpose": "English casual-chat supervised fine-tuning",
        "input_messages": len(all_rows),
        "input_threads": len(threads),
        "excluded_mixed_threads": len(exclusion_manifest),
        "excluded_mixed_messages": sum(item["messages"] for item in exclusion_manifest),
        "usable_threads": len(per_thread),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "validation_split": "whole conversation",
        "validation_thread": validation_thread,
        "filter_counts": dict(sorted(filter_stats.items())),
        "language_audit": language_audit,
        "privacy": {
            "input": "sanitized JSONL only",
            "raw_text_in_report": False,
            "manual_review_still_recommended": True,
        },
    }
    _atomic_private_write(casual_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")

    ideas = idea_records()
    _atomic_private_write(linkedin_dir / "idea_briefs.jsonl", jsonl(ideas))
    _atomic_private_write(
        linkedin_dir / "README.md",
        """# LinkedIn v1 idea bank

These records are prompts for future writing, not training targets. They were derived from
the authorized journals and reflections at the level of themes and experiences. They avoid
copying the journals' long-form style into a LinkedIn model.

For each selected idea, draft or generate several posts, edit the best one until it sounds
like you, and save that final version with its brief. Candidate generation is the later step
that may require model API keys. Keep rejected candidates and concise reasons so they can
later become preference/DPO data.
""",
    )
    _atomic_private_write(
        output_dir / "README.md",
        """# v1 prepared data

`casual_chat/` contains English-only prompt/completion examples built from sanitized chats.
The train/validation boundary is by whole conversation. The heavily mixed Tinglish thread
is excluded as a unit because message-level rules cannot reliably separate every code switch.

`linkedin/idea_briefs.jsonl` is an idea bank, not an SFT dataset. The journals are useful as
source material, but their older long-form style should not be taught directly as LinkedIn
output. Human-approved post drafts can become the first LinkedIn training set later.
""",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the private v1 datasets from sanitized inputs")
    parser.add_argument("--redacted-dir", type=Path, default=Path("data/redacted"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/v1"))
    args = parser.parse_args()
    report = build(args.redacted_dir, args.output_dir)
    print(json.dumps({key: report[key] for key in ("train_examples", "validation_examples", "excluded_mixed_messages")}, indent=2))


if __name__ == "__main__":
    main()
