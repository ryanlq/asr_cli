"""Resumable transcription checkpoints.

VAD segmentation is deterministic for fixed input + params, so a segment's
index is a stable key. Each finished segment is appended (fsync'd) to a JSONL
file; on resume we reload completed segments and skip them. The checkpoint is
keyed by (input path, size, mtime, model, VAD params, language, hotwords) so a
changed file or setting starts fresh automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def jobs_dir() -> str:
    from .download import cache_dir
    return f"{cache_dir()}/jobs"


def job_id(input_path: str, size: int, mtime: int, model_id: str,
           max_segment_sec: float, language: str, hotwords: str) -> str:
    h = hashlib.sha1()
    h.update(os.path.abspath(input_path).encode())
    h.update(f"|{size}|{mtime}|{model_id}|{max_segment_sec}|{language}|{hotwords}".encode())
    return h.hexdigest()[:16]


def ckpt_path(jid: str) -> Path:
    return Path(jobs_dir()) / f"{jid}.jsonl"


def load(path: Path, expected_n: int) -> dict | None:
    """Return {i: (start_ms, end_ms, text)} if valid for expected_n, else None."""
    if not path.exists():
        return None
    done: dict[int, tuple[int, int, str]] = {}
    header_n: int | None = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("t") == "header":
                    header_n = rec.get("n")
                elif "i" in rec:
                    done[int(rec["i"])] = (int(rec["start"]), int(rec["end"]), rec.get("text", ""))
    except (OSError, json.JSONDecodeError):
        return None
    if header_n is not None and header_n != expected_n:
        return None  # stale — params changed
    return done


class Writer:
    """Append-only checkpoint writer. fsync after every segment so a hard kill
    (OOM, SIGKILL) still leaves completed segments on disk."""

    def __init__(self, path: Path, n_segments: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        is_new = (not path.exists()) or path.stat().st_size == 0
        self.f = open(path, "a", encoding="utf-8")
        if is_new:
            self._emit({"t": "header", "n": n_segments})

    def _emit(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def save(self, i: int, start_ms: int, end_ms: int, text: str):
        self._emit({"i": i, "start": start_ms, "end": end_ms, "text": text})

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def discard(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass
