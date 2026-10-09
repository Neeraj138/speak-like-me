import unittest

from speak_like_me.dataset import (
    build_examples,
    clean_message,
    excessive_redaction,
    merge_turns,
    select_validation_thread,
)


def row(order: int, role: str, text: str) -> dict[str, object]:
    return {
        "source": "whatsapp",
        "thread_id": "thread_1",
        "order": order,
        "author_role": role,
        "speaker": "<SELF>" if role == "self" else "<PERSON_1>",
        "text": text,
    }


class DatasetTests(unittest.TestCase):
    def test_merges_consecutive_messages_and_drops_media(self) -> None:
        turns = merge_turns(
            [row(1, "other", "hello"), row(2, "other", "how are you"), row(3, "self", "<Media omitted>")]
        )
        self.assertEqual([(t.role, t.text) for t in turns], [("user", "hello\nhow are you")])

    def test_builds_assistant_target_with_prior_context(self) -> None:
        examples, stats = build_examples(
            [row(1, "other", "are you free later?"), row(2, "self", "yeah, after six works")]
        )
        self.assertEqual(stats["examples"], 1)
        self.assertEqual(examples[0]["prompt"][-1]["role"], "user")
        self.assertEqual(examples[0]["completion"][0]["content"], "yeah, after six works")

    def test_language_turn_resets_context(self) -> None:
        examples, _ = build_examples(
            [
                row(1, "other", "are you free?"),
                row(2, "self", "em cestunnav"),
                row(3, "self", "yes"),
                row(4, "other", "what about tomorrow?"),
                row(5, "self", "tomorrow works"),
            ]
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["prompt"][-1]["content"], "what about tomorrow?")

    def test_redaction_ratio_and_media_cleaning(self) -> None:
        self.assertTrue(excessive_redaction("<PERSON_1> <URL> call"))
        self.assertEqual(clean_message("okay <Media omitted>"), "okay")

    def test_split_selects_whole_thread_nearest_target(self) -> None:
        self.assertEqual(select_validation_thread({"a": 80, "b": 15, "c": 5}), "b")


if __name__ == "__main__":
    unittest.main()
