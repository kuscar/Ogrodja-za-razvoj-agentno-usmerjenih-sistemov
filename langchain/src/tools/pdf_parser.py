from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class PDFParserArgs(BaseModel):
    path: str = Field(..., description="Absolute path to PDF or DOCX file")


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MAX_LINE_LEN = 2000


def _sanitize(text: str) -> str:
    text = _CONTROL_RE.sub("", text)
    return "\n".join(line[:_MAX_LINE_LEN] for line in text.splitlines())


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _parse_docx(path: Path) -> str:
    import docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


@tool("pdf_parser_tool", args_schema=PDFParserArgs)
def pdf_parser_tool(path: str) -> str:
    """Extract sanitized plain text from a CV file (PDF or DOCX)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    head = p.read_bytes()[:4]
    if head.startswith(b"%PDF"):
        text = _parse_pdf(p)
    elif head.startswith(b"PK"):  
        text = _parse_docx(p)
    else:
        raise ValueError(f"Unsupported file type at {path!r}")

    return _sanitize(text)
