import unittest

from training.test_sft_adapter import SYSTEM_PROMPT, normalize_history


class SftInferenceTests(unittest.TestCase):
    def test_one_turn_adds_system_and_user_message(self) -> None:
        messages = normalize_history(None, "are you coming?")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "are you coming?"})

    def test_history_is_not_mutated_or_given_duplicate_system_prompt(self) -> None:
        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hey"},
        ]
        messages = normalize_history(history, "free now?")
        self.assertEqual(len(history), 3)
        self.assertEqual(sum(item["role"] == "system" for item in messages), 1)
        self.assertEqual(messages[-1]["content"], "free now?")


if __name__ == "__main__":
    unittest.main()
