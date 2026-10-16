"""SQLite 持久化 → TF-IDF 检索 → DeepSeek 生成回答。"""

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from rag.documents import parse_document
from rag.llm import GenerationError, ModelConfig, generate_answer

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "knowledge.db"


def _is_title_request(question: str) -> bool:
    """识别简单的文档标题/列表查询，避免将元数据请求当成正文关键词。"""
    text = re.sub(r"[\s，,。.!！?？：:；;、]", "", question)
    text = re.sub(
        r"列出来|告诉我|看一下|是什么|叫什么|有哪些|请|麻烦|帮我|给我|发我|"
        r"列出|查看|看下|看看|显示|提供|把|将|是啥|一下|呢|吧", "", text,
    )
    text = re.sub(
        r"知识库[中里内]?|已上传的?|已导入的?|上传的|导入的|当前|这个|这份|这些|所有|全部|的",
        "", text,
    )
    return text in {"标题", "文档标题", "文件标题", "文件名", "文档名", "文档名称",
                    "文件名称", "文档列表", "文件列表"}


@dataclass(frozen=True)
class SearchHit:
    source: str
    text: str
    score: float
    chunk_number: int
    page: int | None


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchHit]
    generated: bool = False
    warning: str | None = None


class KnowledgeBase:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    digest TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    UNIQUE(document_id, position)
                );
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_document(self, filename: str, content: bytes) -> tuple[str, int]:
        """返回（新增/更新/未变化，分块数）；解析失败不会覆盖原文档。"""
        name, chunks = parse_document(filename, content)
        digest = hashlib.sha256(content).hexdigest()
        with self._connect() as db:
            old = db.execute("SELECT id, digest FROM documents WHERE name = ?", (name,)).fetchone()
            if old and old["digest"] == digest:
                count = db.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (old["id"],)
                ).fetchone()[0]
                return "未变化", count
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if old:
                document_id = old["id"]
                db.execute("UPDATE documents SET digest = ?, updated_at = ? WHERE id = ?",
                           (digest, now, document_id))
                db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            else:
                cursor = db.execute("INSERT INTO documents(name, digest, updated_at) VALUES (?, ?, ?)",
                                    (name, digest, now))
                document_id = cursor.lastrowid
            db.executemany(
                "INSERT INTO chunks(document_id, position, text, page) VALUES (?, ?, ?, ?)",
                [(document_id, i, chunk.text, chunk.page) for i, chunk in enumerate(chunks, 1)],
            )
        return "更新" if old else "新增", len(chunks)

    def list_documents(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""
                SELECT d.id, d.name, d.updated_at, COUNT(c.id) AS chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id ORDER BY d.name
            """).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: int) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def _document_titles(self) -> Answer:
        with self._connect() as db:
            documents = db.execute("""
                SELECT d.name, c.text FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id AND c.position = 1
                ORDER BY d.name
            """).fetchall()
        if not documents:
            return Answer("知识库还没有文档，请先上传并点击「导入知识库」。", [])
        lines = ["当前知识库的文档标题 / 文件名：", ""]
        for i, document in enumerate(documents, 1):
            # 一级标题作为 Markdown 文档标题；其他文件按文件名显示，不猜测正文标题。
            heading = None
            if Path(document["name"]).suffix.lower() == ".md":
                heading = re.search(
                    r"^#[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", document["text"] or "", re.MULTILINE
                )
            if heading:
                lines.append(f"{i}. 标题：{heading.group(1)}（文件：{document['name']}）")
            else:
                lines.append(f"{i}. 文件名：{document['name']}")
        return Answer("\n".join(lines), [])

    def search(self, question: str, top_k: int = 3, min_score: float = 0.05) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间。")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score 必须在 0 到 1 之间。")
        question = re.sub(r"\s+", " ", question).strip()
        if not question:
            return []
        with self._connect() as db:
            rows = db.execute("""
                SELECT d.name, c.text, c.position, c.page FROM chunks c
                JOIN documents d ON d.id = c.document_id ORDER BY c.id
            """).fetchall()
        if not rows:
            return []
        # 字符 2～4 gram 无需中文分词或下载模型；默认 L2 归一化后的点积等于余弦相似度。
        # 简单项目在提问时重建内存索引，确保重启、更新和删除后的检索始终读取当前数据。
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True,
                                     max_features=100_000)
        matrix = vectorizer.fit_transform([re.sub(r"\s+", " ", row["text"]) for row in rows])
        query = vectorizer.transform([question])
        scores = (matrix @ query.T).toarray().ravel()
        indices = scores.argsort()[::-1]
        return [
            SearchHit(rows[i]["name"], rows[i]["text"], float(scores[i]),
                      rows[i]["position"], rows[i]["page"])
            for i in indices[:top_k]
            if scores[i] > 0 and scores[i] >= min_score
        ]

    def ask(
        self, question: str, config: ModelConfig, top_k: int = 3, min_score: float = 0.05
    ) -> Answer:
        if not question.strip():
            raise ValueError("请输入问题。")
        if len(question) > 2000:
            raise ValueError("问题不能超过 2000 个字符。")
        if _is_title_request(question):
            return self._document_titles()
        hits = self.search(question, top_k, min_score)
        if not hits:
            return Answer("知识库中没有找到相关内容。请补充文档、换一种问法或适当降低检索阈值。", [])
        context = json.dumps([
            {"编号": i, "文件": hit.source, "页码": hit.page,
             "片段": hit.chunk_number, "内容": hit.text}
            for i, hit in enumerate(hits, 1)
        ], ensure_ascii=False)
        try:
            return Answer(generate_answer(question, context, config), hits, generated=True)
        except GenerationError as exc:
            return Answer(
                "以下是检索到的知识库原文（尚未生成模型回答）：\n\n" + "\n\n".join(
                    f"[{i}] {hit.source}\n\n{hit.text}" for i, hit in enumerate(hits, 1)
                ), hits, warning=str(exc),
            )
