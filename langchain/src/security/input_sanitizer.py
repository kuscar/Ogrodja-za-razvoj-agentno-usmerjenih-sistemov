from __future__ import annotations

import re
import unicodedata

_BAD_CHARS = re.compile(r"[​-‏‪-‮﻿]")  


def sanitize_user_text(text: str, max_len: int = 60_000) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _BAD_CHARS.sub("", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return text[:max_len]
