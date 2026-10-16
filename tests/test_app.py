import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from rag.engine import KnowledgeBase
from rag.llm import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.kb = KnowledgeBase(Path(temp.name) / "test.db")
        factory = patch("rag.engine.KnowledgeBase", return_value=self.kb)
        factory.start()
        self.addCleanup(factory.stop)
        config = patch("rag.llm.ModelConfig.from_env", return_value=ModelConfig())
        config.start()
        self.addCleanup(config.stop)
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def test_empty_library_sample_import_question_and_delete(self):
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(self.app.chat_input[0].disabled)

        self.button("导入示例文档").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.kb.list_documents()), 1)
        self.assertFalse(self.app.chat_input[0].disabled)
        self.app.chat_input[0].set_value("年假有多少天？").run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertTrue(any("10 天" in item.value for item in self.app.markdown))
        self.assertTrue(any("API_KEY" in warning.value for warning in self.app.warning))
        self.button("清空对话").click().run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.button("删除选中文档").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.kb.list_documents(), [])
        self.assertTrue(self.app.chat_input[0].disabled)

    @patch("rag.engine.generate_answer", return_value="员工每年享有 10 天带薪年假。[1]")
    def test_generated_answer_and_sources_are_displayed(self, generate):
        self.app.run()
        self.button("导入示例文档").click().run()
        self.app.chat_input[0].set_value("年假有多少天？").run()
        self.assertEqual(len(self.app.exception), 0)
        generate.assert_called_once()
        self.assertTrue(any("10 天带薪年假。[1]" in item.value for item in self.app.markdown))
        self.assertTrue(any("员工入职满一年" in item.value for item in self.app.text))

    @patch("rag.engine.generate_answer")
    def test_title_question_returns_document_title_without_key(self, generate):
        self.app.run()
        self.button("导入示例文档").click().run()
        self.app.chat_input[0].set_value("标题给我").run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(any("标题：星河科技员工手册（虚构示例）" in item.value
                            for item in self.app.markdown))
        self.assertEqual(len(self.app.warning), 0)
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
