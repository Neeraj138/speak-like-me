import unittest

from speak_like_me.review import prepare_review_records, review_focus, select_review_sample


def example(index: int, target: str, prompt: str = "hello") -> dict:
    return {
        "example_id": f"id_{index}",
        "prompt": [{"role": "system", "content": "reply"}, {"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": target}],
        "metadata": {"source": "whatsapp", "thread_id": "thread", "target_order": index},
    }


class ReviewTests(unittest.TestCase):
    def test_sample_is_deterministic_unique_and_requested_size(self) -> None:
        rows = [example(i, "x" * (i + 1), "hello <PERSON_1>" if i % 7 == 0 else "hello") for i in range(60)]
        first = select_review_sample(rows, 25, seed="fixed")
        second = select_review_sample(rows, 25, seed="fixed")
        self.assertEqual([x["example_id"] for x in first], [x["example_id"] for x in second])
        self.assertEqual(len(first), 25)
        self.assertEqual(len({x["example_id"] for x in first}), 25)

    def test_focus_marks_review_risks(self) -> None:
        self.assertIn("redaction_placeholder", review_focus(example(1, "ok", "ask <PERSON_1>")))
        self.assertIn("very_short_reply", review_focus(example(1, "ok")))
        self.assertIn("long_reply", review_focus(example(1, "x" * 301)))

    def test_review_records_keep_split_counts(self) -> None:
        train = [example(i, "train reply") for i in range(20)]
        validation = [example(i + 100, "validation reply") for i in range(10)]
        rows = prepare_review_records(train, validation, train_count=12, validation_count=5)
        self.assertEqual(sum(x["split"] == "train" for x in rows), 12)
        self.assertEqual(sum(x["split"] == "validation" for x in rows), 5)


if __name__ == "__main__":
    unittest.main()
