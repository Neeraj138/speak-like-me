import unittest

from speak_like_me.preference import select_preference_prompts, to_preference_record


def example(index: int, length: int) -> dict:
    return {
        "example_id": f"example_{index}",
        "prompt": [{"role": "user", "content": "hello"}],
        "completion": [{"role": "assistant", "content": "x" * length}],
        "metadata": {"source": "whatsapp"},
    }


class PreferenceTests(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_balanced(self) -> None:
        rows = [example(i, 10 if i < 50 else 80 if i < 150 else 250) for i in range(200)]
        first = select_preference_prompts(rows, 100)
        second = select_preference_prompts(rows, 100)
        self.assertEqual([x["example_id"] for x in first], [x["example_id"] for x in second])
        self.assertEqual(len({x["example_id"] for x in first}), 100)
        lengths = [len(x["completion"][0]["content"]) for x in first]
        self.assertEqual(sum(x <= 20 for x in lengths), 30)
        self.assertEqual(sum(20 < x <= 160 for x in lengths), 50)
        self.assertEqual(sum(x > 160 for x in lengths), 20)

    def test_record_preserves_reference_as_private_candidate_source(self) -> None:
        record = to_preference_record(example(1, 5), 1)
        self.assertEqual(record["preference_id"], "casual_pref_001")
        self.assertEqual(record["reference_completion"], "xxxxx")


if __name__ == "__main__":
    unittest.main()
