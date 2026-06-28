"""Output formatters. A segment is a (start_ms, end_ms, text) tuple."""

from __future__ import annotations

import json


def to_text(segments) -> str:
    return "".join(t for _, _, t in segments).strip()


def to_srt(segments) -> str:
    lines = []
    idx = 1
    for start, end, text in segments:
        text = text.strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_ms(start)} --> {_ms(end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines) + ("\n" if lines else "")


def to_json(segments) -> str:
    items = [{"text": t.strip(), "start": s, "end": e}
             for s, e, t in segments if t.strip()]
    return json.dumps(items, ensure_ascii=False, indent=2) + "\n"


def _ms(ms: int) -> str:
    ms = int(ms)
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1_000
    millis = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"
