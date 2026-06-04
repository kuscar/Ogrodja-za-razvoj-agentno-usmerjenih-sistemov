from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field


DANGEROUS_TOKENS = [
    r"\\write18",
    r"\\immediate\\write",
    r"\\openout",
    r"\\input\s*\{[^}]*\.\.",
    r"\\catcode",
    r"\\read",
    r"\\input\s*\{/",
]


class LatexValidatorArgs(BaseModel):
    latex_source: str = Field(..., description="The LaTeX source to validate")
    user_id: str = Field(..., description="Tenant id — used to namespace output dir")
    friendly_name: str = Field("cv", description="Human-readable stem for the output PDF filename")


def _static_check(src: str) -> list[str]:
    errors = []
    for pattern in DANGEROUS_TOKENS:
        if re.search(pattern, src):
            errors.append(f"disallowed token matching /{pattern}/")
    return errors


def _safe_stem(name: str) -> str:
    """Turn an arbitrary string into a safe, lowercase filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:60] or "cv"


@tool("latex_validator_tool", args_schema=LatexValidatorArgs)
def latex_validator_tool(latex_source: str, user_id: str, friendly_name: str = "cv") -> dict:
    """Validate then compile a LaTeX source string."""
    latex_source = "".join(
        ch for ch in latex_source if ord(ch) >= 0x20 or ch in "\t\n\r"
    )

    static_errors = _static_check(latex_source)
    if static_errors:
        return {"ok": False, "pdf_path": None, "errors": static_errors}

    with tempfile.TemporaryDirectory(prefix=f"cv_{user_id}_") as tmp:
        tmp_dir = Path(tmp)
        tex_path = tmp_dir / "cv.tex"
        tex_path.write_text(latex_source, encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-no-shell-escape",
                    "-output-directory",
                    str(tmp_dir),
                    str(tex_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "pdf_path": None, "errors": ["pdflatex timeout"]}

        if proc.returncode != 0:
            tail = proc.stdout[-1200:] if proc.stdout else proc.stderr[-1200:]
            return {"ok": False, "pdf_path": None, "errors": [tail]}

        out_dir = Path("/var/cv-artifacts") / user_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(friendly_name)
        final = out_dir / f"{stem}_{uuid.uuid4().hex[:8]}.pdf"
        shutil.move(tmp_dir / "cv.pdf", final)
        return {"ok": True, "pdf_path": str(final), "pdf_name": final.name, "errors": []}
