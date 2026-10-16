"""文档解析与重叠分块；PDF 保留页码，方便回答溯源。"""

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import PureWindowsPath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

MAX_FILE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int | None = None


def split_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """按字符分块，优先在后半段的句末或换行处切开。"""
    if chunk_size < 2 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size 必须至少为 2，且 0 <= overlap < chunk_size。")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 切点必须超过 overlap，保证下一轮能够向前推进。
            lower = start + max(chunk_size // 2, overlap + 1)
            boundaries = list(re.finditer(r"[\n。！？.!?；;]", text[lower:end]))
            if boundaries:
                end = lower + boundaries[-1].end()
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _decode_text(content: bytes) -> str:
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码无法识别，请将文件另存为 UTF-8 后上传。")


def parse_document(
    filename: str, content: bytes, chunk_size: int = 600, overlap: int = 100
) -> tuple[str, list[TextChunk]]:
    """从上传内容解析文本，不将用户提供的路径写入文件系统。"""
    name = PureWindowsPath(filename).name
    if not name or not content:
        raise ValueError("文件名称或内容为空。")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("单个文件不能超过 20 MB。")
    extension = PureWindowsPath(name).suffix.lower()
    pages: list[tuple[int | None, str]] = []
    if extension in {".txt", ".md"}:
        pages.append((None, _decode_text(content)))
    elif extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF 已加密，请解密后再上传。")
            pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        except (PyPdfError, OSError, UnicodeError) as exc:
            raise ValueError("PDF 解析失败，请检查文件是否损坏。") from exc
    else:
        raise ValueError("仅支持 TXT、Markdown（.md）和 PDF 文件。")
    chunks = [
        TextChunk(chunk, page)
        for page, text in pages
        for chunk in split_text(text, chunk_size, overlap)
        if len(chunk) >= 2
    ]
    if not chunks:
        raise ValueError("没有可索引的文本。扫描版 PDF 需要先做 OCR，文本至少需要两个字符。")
    return name, chunks
