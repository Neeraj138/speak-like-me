import unittest

from speak_like_me.language import contains_telugu_or_mixed


class LanguageTests(unittest.TestCase):
    def test_recognizes_romanized_telugu(self) -> None:
        self.assertTrue(contains_telugu_or_mixed("em cestunnav"))
        self.assertTrue(contains_telugu_or_mixed("nuvvu ekkada unnava"))
        self.assertTrue(contains_telugu_or_mixed("Aa background music entiii??"))
        self.assertTrue(contains_telugu_or_mixed("try cheyyu install ayyaka"))
        self.assertTrue(contains_telugu_or_mixed("Sunday kuda open cheskovachu"))

    def test_does_not_reject_normal_english(self) -> None:
        self.assertFalse(contains_telugu_or_mixed("what are you doing"))
        self.assertFalse(contains_telugu_or_mixed("sure, I can check this today"))


if __name__ == "__main__":
    unittest.main()
