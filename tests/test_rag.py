import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag.documents import parse_document, split_text
from rag.engine import KnowledgeBase
from rag.llm import API_URL, GenerationError, ModelConfig, generate_answer


class DocumentTests(unittest.TestCase):
    def test_chunking_preserves_characters_and_overlap(self):
        text = "0123456789" * 150
        chunks = split_text(text, chunk_size=100, overlap=20)
        rebuilt = chunks[0] + "".join(chunk[20:] for chunk in chunks[1:])
        self.assertEqual(rebuilt, text)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        with self.assertRaises(ValueError):
            split_text(text, chunk_size=100, overlap=100)

    def test_chinese_encodings_and_filename(self):
        for encoding in ("utf-8-sig", "gb18030", "utf-16"):
            name, chunks = parse_document("../../手册.txt", "每年有十天年假。".encode(encoding))
            self.assertEqual(name, "手册.txt")
            self.assertIn("十天年假", chunks[0].text)

    def test_pdf_retains_page_number(self):
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 30 250 Td (Annual leave is 10 days.) Tj ET")
        page[NameObject("/Contents")] = stream
        output = BytesIO()
        writer.write(output)
        _, chunks = parse_document("leave.pdf", output.getvalue())
        self.assertEqual(chunks[0].page, 2)
        self.assertIn("10 days", chunks[0].text)

    def test_invalid_and_empty_documents(self):
        for filename, data in (("x.txt", b""), ("x.txt", b"   "),
                               ("x.docx", b"hello"), ("x.pdf", b"not a PDF")):
            with self.subTest(filename=filename, data=data), self.assertRaises(ValueError):
                parse_document(filename, data)


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "knowledge.db"
        self.kb = KnowledgeBase(self.db)

    def add(self, filename, text):
        return self.kb.add_document(filename, text.encode("utf-8"))

    def test_chinese_retrieval_and_restart(self):
        self.add("年假.md", "员工每年享有十天带薪年假，需要提前三天申请。")
        self.add("设备.md", "办公电脑出现故障请联系技术支持处理。")
        restarted = KnowledgeBase(self.db)
        hits = restarted.search("带薪年假有多少天？")
        self.assertEqual(hits[0].source, "年假.md")
        self.assertIn("十天", hits[0].text)
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(len(restarted.list_documents()), 2)

    def test_deduplicate_update_and_delete(self):
        self.assertEqual(self.add("手册.md", "员工每年有十天年假。")[0], "新增")
        self.assertEqual(self.add("手册.md", "员工每年有十天年假。")[0], "未变化")
        self.assertEqual(self.add("手册.md", "员工每年有十五天年假。")[0], "更新")
        self.assertEqual(len(self.kb.list_documents()), 1)
        self.assertIn("十五天", self.kb.search("员工年假")[0].text)
        self.kb.delete_document(self.kb.list_documents()[0]["id"])
        self.assertEqual(self.kb.list_documents(), [])
        self.assertEqual(self.kb.search("员工年假"), [])

    def test_invalid_replacement_keeps_original(self):
        self.add("手册.md", "员工享有十天带薪年假。")
        with self.assertRaises(ValueError):
            self.kb.add_document("手册.md", b"   ")
        self.assertIn("十天", self.kb.search("年假")[0].text)

    @patch("rag.engine.generate_answer")
    def test_no_match_does_not_call_api(self, generate):
        self.add("手册.md", "员工享有十天带薪年假。")
        result = self.kb.ask("quantum astronomy", ModelConfig("fake-key"))
        self.assertFalse(result.generated)
        self.assertEqual(result.sources, [])
        generate.assert_not_called()

    def test_missing_key_returns_explicit_retrieval_result(self):
        self.add("手册.md", "员工享有十天带薪年假。")
        result = self.kb.ask("带薪年假", ModelConfig())
        self.assertFalse(result.generated)
        self.assertIn("API_KEY", result.warning)
        self.assertIn("尚未生成", result.text)
        self.assertEqual(result.sources[0].source, "手册.md")

    @patch("rag.engine.generate_answer", return_value="员工有十天年假。[1]")
    def test_rag_sends_question_and_numbered_context(self, generate):
        self.add("手册.md", "员工享有十天带薪年假。")
        result = self.kb.ask("带薪年假", ModelConfig("fake-key"))
        self.assertTrue(result.generated)
        self.assertIn("[1]", result.text)
        question, context, _ = generate.call_args.args
        self.assertEqual(question, "带薪年假")
        self.assertEqual(json.loads(context)[0]["编号"], 1)
        self.assertIn("十天", json.loads(context)[0]["内容"])

    @patch("rag.engine.generate_answer")
    def test_document_title_requests_bypass_body_search_and_api(self, generate):
        self.add("手册.md", "# 星河科技员工手册\n\n员工享有十天带薪年假。")
        self.add("联系方式.txt", "办公电脑出现故障请联系技术支持。")
        for question in ("标题给我", "给我标题", "这份文档的标题是什么？",
                         "请列出知识库里的文件名", "文档列表"):
            with self.subTest(question=question):
                result = self.kb.ask(question, ModelConfig(), min_score=0.8)
                self.assertIn("星河科技员工手册", result.text)
                self.assertIn("手册.md", result.text)
                self.assertIn("联系方式.txt", result.text)
                self.assertIsNone(result.warning)
        generate.assert_not_called()

    @patch("rag.engine.generate_answer")
    def test_title_request_with_empty_library(self, generate):
        result = self.kb.ask("标题给我", ModelConfig())
        self.assertIn("还没有文档", result.text)
        generate.assert_not_called()

    @patch("rag.engine.generate_answer", return_value="需要直属主管审批。[1]")
    def test_content_question_containing_title_is_still_retrieved(self, generate):
        self.add("手册.md", "# 员工手册\n\n修改文档标题需要直属主管审批。")
        result = self.kb.ask("修改文档标题需要谁审批？", ModelConfig("fake-key"))
        self.assertTrue(result.generated)
        generate.assert_called_once()


class ApiTests(unittest.TestCase):
    @patch("rag.llm.requests.post")
    def test_deepseek_payload_and_response(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": [{"message": {"content": "十天。[1]"}}]}
        result = generate_answer("年假？", "[]", ModelConfig("fake-key", "my-model"))
        self.assertEqual(result, "十天。[1]")
        args, kwargs = post.call_args
        self.assertEqual(args[0], API_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer fake-key")
        self.assertEqual(kwargs["json"]["model"], "my-model")
        self.assertFalse(kwargs["json"]["stream"])
        self.assertIn("年假？", kwargs["json"]["messages"][-1]["content"])
        self.assertNotIn("fake-key", repr(ModelConfig("fake-key")))

    @patch("rag.llm.requests.post")
    def test_api_failure_is_readable_and_redacted(self, post):
        for status in (401, 402, 429, 500):
            post.return_value = Mock(status_code=status)
            with self.subTest(status=status), self.assertRaises(GenerationError) as error:
                generate_answer("问题", "[]", ModelConfig("secret-value"))
            self.assertIn(str(status), str(error.exception))
            self.assertNotIn("secret-value", str(error.exception))

    @patch("rag.llm.requests.post")
    def test_timeout_and_malformed_response(self, post):
        post.side_effect = requests.Timeout()
        with self.assertRaisesRegex(GenerationError, "超时"):
            generate_answer("问题", "[]", ModelConfig("fake-key"))
        post.side_effect = None
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": []}
        with self.assertRaisesRegex(GenerationError, "格式异常"):
            generate_answer("问题", "[]", ModelConfig("fake-key"))


if __name__ == "__main__":
    unittest.main()
