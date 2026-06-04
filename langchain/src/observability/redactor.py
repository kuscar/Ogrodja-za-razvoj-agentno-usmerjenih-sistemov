from __future__ import annotations

import hashlib
import re
from typing import Any

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d[\d \-().]{7,}\d")
JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")
OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{20,}")
ANTHROPIC_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

SCRUB_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (EMAIL_RE, "<redacted:email>"),
    (JWT_RE, "<redacted:jwt>"),
    (OPENAI_KEY_RE, "<redacted:openai_key>"),
    (ANTHROPIC_KEY_RE, "<redacted:anthropic_key>"),
    (PHONE_RE, "<redacted:phone>"),
]

BLOCKED_KEYS: set[str] = {
    "raw_cv_path",
    "enrichment_text",
    "job_description",
    "structured_profile",
    "latex_source",
    "cover_letter",
    "summary",
    "full_name",
    "email",
    "phone",
    "location",
    "bullets",
    "experiences",
    "education",
    "embedding",
    "payload",
    "Authorization",
    "authorization",
    "credentials",
    "token",
    "service_role_key",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
}

BLOCKED_KEY_RE = re.compile(r"(_raw|_text|_body|_content|_secret|_key|password)$", re.I)


def hash_user_id(user_id: str, salt: str = "cv-builder") -> str:
    """Stable, short, irreversible tag we can put in logs."""
    if not user_id:
        return "anon"
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return f"u:{digest[:10]}"


def _scrub_string(s: str) -> str:
    for pattern, replacement in SCRUB_PATTERNS:
        s = pattern.sub(replacement, s)
    return s


def redact(obj: Any, *, depth: int = 0) -> Any:
    """Walk the object tree, dropping blocked fields and scrubbing strings."""
    if depth > 6:
        return "<redacted:depth>"

    if isinstance(obj, str):
        return _scrub_string(obj)

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in BLOCKED_KEYS or BLOCKED_KEY_RE.search(k):
                out[k] = f"<redacted:{k}>"
                continue
            if k == "user_id" and isinstance(v, str):
                out[k] = hash_user_id(v)
                continue
            out[k] = redact(v, depth=depth + 1)
        return out

    if isinstance(obj, (list, tuple, set)):
        return type(obj)(redact(v, depth=depth + 1) for v in obj)

    return obj
