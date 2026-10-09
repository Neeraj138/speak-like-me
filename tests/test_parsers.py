from pathlib import Path
import tempfile
import unittest
import zipfile

from speak_like_me.parsers import (
    parse_linkedin,
    parse_telegram,
    parse_whatsapp,
    parse_writings,
    thread_summaries,
)


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "examples" / "synthetic"


class ParserTests(unittest.TestCase):
    def test_whatsapp_parses_multiline_and_speakers(self) -> None:
        parsed = parse_whatsapp(SYNTHETIC / "whatsapp_sample.txt")
        self.assertEqual(len(parsed.messages), 4)
        self.assertEqual(parsed.messages[-1].sender, "Demo User")
        self.assertIn("\nthis is a multiline message", parsed.messages[-1].text)

    def test_whatsapp_ignores_service_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.txt"
            path.write_text(
                "12/09/2026, 08:00 - Messages are end-to-end encrypted.\n"
                "12/09/2026, 08:01 - Demo User: hello\n",
                encoding="utf-8",
            )
            parsed = parse_whatsapp(path)
        self.assertEqual([message.text for message in parsed.messages], ["hello"])

    def test_whatsapp_reads_zip_without_extracting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "WhatsApp Chat.txt",
                    "12/09/2026, 08:01 - Demo User: hello from zip\n",
                )
            parsed = parse_whatsapp(path)
            self.assertEqual([message.text for message in parsed.messages], ["hello from zip"])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_telegram_flattens_rich_text(self) -> None:
        parsed = parse_telegram(SYNTHETIC / "telegram_sample.json")
        self.assertEqual(len(parsed.messages), 2)
        self.assertEqual(parsed.messages[1].text, "sure, ill be there")
        self.assertEqual(thread_summaries(parsed)[0]["selection_id"], "123456")

    def test_telegram_html_joins_continuations_and_omits_forwards(self) -> None:
        html = """
        <div class="message default clearfix" id="message1">
          <div class="body"><div class="pull_right date details" title="date one"></div>
          <div class="from_name">Demo User</div><div class="text">first</div></div>
        </div>
        <div class="message default clearfix joined" id="message2">
          <div class="body"><div class="text">second<br>line</div></div>
        </div>
        <div class="message default clearfix" id="message3">
          <div class="body"><div class="from_name">Demo User</div>
          <div class="forwarded body"><div class="from_name">Private Sender</div>
          <div class="text">forwarded private text</div></div></div>
        </div>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messages.html"
            path.write_text(html, encoding="utf-8")
            parsed = parse_telegram(path)
        self.assertEqual([message.text for message in parsed.messages], ["first", "second\nline"])
        self.assertEqual({message.sender for message in parsed.messages}, {"Demo User"})

    def test_writings_reads_docx_paragraphs(self) -> None:
        document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body><w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>
          <w:p><w:r><w:t>Second</w:t><w:tab/><w:t>part</w:t></w:r></w:p></w:body>
        </w:document>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            parsed = parse_writings(path)
        self.assertEqual(parsed.messages[0].text, "First paragraph\n\nSecond\tpart")

    def test_linkedin_preserves_selection_id_only_in_private_metadata(self) -> None:
        parsed = parse_linkedin(SYNTHETIC / "linkedin_messages_sample.csv")
        self.assertEqual(len(parsed.messages), 3)
        self.assertEqual(parsed.messages[0].metadata["selection_id"], "conversation-one")
        self.assertTrue(parsed.messages[0].source_thread_id.startswith("linkedin_"))


if __name__ == "__main__":
    unittest.main()
