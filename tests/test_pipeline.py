import json
from pathlib import Path
import tempfile
import unittest

from speak_like_me.config import RedactionConfig
from speak_like_me.pipeline import audit_sanitized, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "examples" / "synthetic"
CONFIG = RedactionConfig.from_file(SYNTHETIC / "config.json")


class PipelineTests(unittest.TestCase):
    def test_whatsapp_pipeline_writes_private_sanitized_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sanitized.jsonl"
            report_path = Path(directory) / "report.json"
            report = run_pipeline(
                source="whatsapp",
                input_path=SYNTHETIC / "whatsapp_sample.txt",
                output_path=output,
                report_path=report_path,
                config=CONFIG,
            )
            content = output.read_text(encoding="utf-8")
            audit = audit_sanitized(output)

            self.assertEqual(report["output_messages"], 4)
            self.assertNotIn("Alex Example", content)
            self.assertNotIn("example.test", content)
            self.assertIn("<EMAIL>", content)
            self.assertTrue(audit["passed"])
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(report_path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(json.loads(report_path.read_text())["contains_raw_text"])

    def test_strict_mode_redacts_long_spaced_number_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.txt"
            raw.write_text(
                "[14/09/2026, 9:10 AM] Demo User: suspicious 12 34 56 78 90 12 34 56 78 90\n",
                encoding="utf-8",
            )
            output = Path(directory) / "sanitized.jsonl"
            report = Path(directory) / "report.json"
            run_pipeline(
                source="whatsapp",
                input_path=raw,
                output_path=output,
                report_path=report,
                config=CONFIG,
            )
            self.assertNotIn("12 34 56 78", output.read_text(encoding="utf-8"))
            self.assertTrue(audit_sanitized(output)["passed"])

    def test_linkedin_requires_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sanitized.jsonl"
            report = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "requires at least one"):
                run_pipeline(
                    source="linkedin",
                    input_path=SYNTHETIC / "linkedin_messages_sample.csv",
                    output_path=output,
                    report_path=report,
                    config=CONFIG,
                )

    def test_linkedin_selection_excludes_other_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sanitized.jsonl"
            report_path = Path(directory) / "report.json"
            run_pipeline(
                source="linkedin",
                input_path=SYNTHETIC / "linkedin_messages_sample.csv",
                output_path=output,
                report_path=report_path,
                config=CONFIG,
                selected_threads={"conversation-one"},
            )
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertNotIn("Unselected Person", output.read_text())

    def test_multi_chat_telegram_requires_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "result.json"
            raw.write_text(
                json.dumps(
                    {
                        "chats": {
                            "list": [
                                {
                                    "id": 1,
                                    "name": "One",
                                    "messages": [
                                        {"type": "message", "from": "Demo User", "text": "one"}
                                    ],
                                },
                                {
                                    "id": 2,
                                    "name": "Two",
                                    "messages": [
                                        {"type": "message", "from": "Demo User", "text": "two"}
                                    ],
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "multi-chat Telegram"):
                run_pipeline(
                    source="telegram",
                    input_path=raw,
                    output_path=Path(directory) / "sanitized.jsonl",
                    report_path=Path(directory) / "report.json",
                    config=CONFIG,
                )


if __name__ == "__main__":
    unittest.main()
