from pathlib import Path
import unittest

from speak_like_me.config import RedactionConfig
from speak_like_me.detectors import OllamaDetector
from speak_like_me.models import DetectedEntity, Message
from speak_like_me.redaction import Redactor, residual_risks


ROOT = Path(__file__).resolve().parents[1]
CONFIG = RedactionConfig.from_file(ROOT / "examples" / "synthetic" / "config.json")


class FakeDetector:
    def detect(self, text: str) -> list[DetectedEntity]:
        if "private neighborhood" in text:
            return [DetectedEntity("private neighborhood", "LOCATION")]
        return []


class RedactionTests(unittest.TestCase):
    def test_redacts_otp_password_links_cards_and_indian_identifiers(self) -> None:
        redactor = Redactor(RedactionConfig(self_names=("Demo User",)))
        message = Message(
            source="whatsapp",
            source_thread_id="thread",
            order=0,
            sender="Demo User",
            text=(
                "OTP is 482911, password is horse-battery, visit example.com/reset, "
                "card 4111 1111 1111 1111, UPI demo.user@okbank, "
                "PAN ABCDE1234F, IFSC HDFC0123456, Aadhaar 1234 5678 9012"
            ),
        )
        result = redactor.redact_messages([message])[0].text
        self.assertNotIn("482911", result)
        self.assertNotIn("horse-battery", result)
        self.assertNotIn("example.com", result)
        self.assertNotIn("4111", result)
        self.assertNotIn("demo.user@okbank", result)
        self.assertNotIn("ABCDE1234F", result)
        self.assertNotIn("HDFC0123456", result)
        self.assertNotIn("1234 5678 9012", result)
        self.assertIn("<OTP>", result)
        self.assertIn("<SECRET>", result)
        self.assertIn("<URL>", result)
        self.assertIn("<CARD_NUMBER>", result)
        self.assertIn("<UPI_ID>", result)
        self.assertIn("<PAN>", result)
        self.assertIn("<IFSC>", result)
        self.assertIn("<AADHAAR>", result)

    def test_redacts_deterministic_private_data_and_preserves_style(self) -> None:
        message = Message(
            source="whatsapp",
            source_thread_id="thread",
            order=0,
            sender="Demo User",
            text=(
                "yeah ill call +91 98765 43210, email demo@example.test, "
                "use password=hunter2 and card 4111 1111 1111 1111"
            ),
        )
        result = Redactor(CONFIG).redact_messages([message])[0]
        self.assertEqual(result.author_role, "self")
        self.assertEqual(result.speaker, "<SELF>")
        self.assertIn("yeah ill call <PHONE>", result.text)
        self.assertIn("<EMAIL>", result.text)
        self.assertIn("password=<SECRET>", result.text)
        self.assertIn("card <CARD_NUMBER>", result.text)
        self.assertFalse(residual_risks(result.text))

    def test_custom_and_context_entities_are_stable(self) -> None:
        messages = [
            Message("writing", "thread", 0, "__self__", "Project Firefly is in private neighborhood"),
            Message("writing", "thread", 1, "__self__", "private neighborhood is quiet"),
        ]
        results = Redactor(CONFIG, detectors=[FakeDetector()]).redact_messages(messages)
        self.assertIn("<PROJECT_1>", results[0].text)
        self.assertIn("<LOCATION_1>", results[0].text)
        self.assertIn("<LOCATION_1>", results[1].text)

    def test_participant_name_does_not_replace_substrings(self) -> None:
        messages = [
            Message("whatsapp", "thread", 0, "Ann", "an announcement from Ann"),
            Message("whatsapp", "thread", 1, "Demo User", "okay"),
        ]
        result = Redactor(CONFIG).redact_messages(messages)[0]
        self.assertEqual(result.text, "an announcement from <PERSON_1>")

    def test_ollama_rejects_non_loopback_url(self) -> None:
        with self.assertRaises(ValueError):
            OllamaDetector(model="example", base_url="https://remote.example.test")


if __name__ == "__main__":
    unittest.main()
