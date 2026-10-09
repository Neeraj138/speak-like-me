import unittest

from speak_like_me.llamacpp import LlamaCppLanguageFilter, _batches


class LlamaCppTests(unittest.TestCase):
    def test_rejects_non_loopback_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "localhost"):
            LlamaCppLanguageFilter(base_url="https://example.com")

    def test_batches_preserve_all_items(self) -> None:
        items = [(str(index), "abcd") for index in range(5)]
        batches = list(_batches(items, max_items=2, max_characters=100))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual([item for batch in batches for item in batch], items)
