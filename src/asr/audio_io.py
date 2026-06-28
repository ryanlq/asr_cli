"""Audio loading: decode any container/codec to 16 kHz mono f32 via ffmpeg."""

from __future__ import annotations

import subprocess
import sys

import numpy as np


def load_audio(path: str, sr: int = 16000):
    """Return (samples: np.float32 mono, sample_rate). path='-' reads stdin."""
    if path == "-":
        stdin = sys.stdin.buffer.read()
        cmd = ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
               "-ar", str(sr), "-ac", "1", "-f", "s16le", "pipe:1"]
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found — install it (sudo apt install ffmpeg)")
    else:
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", path,
               "-ar", str(sr), "-ac", "1", "-f", "s16le", "pipe:1"]
        try:
            proc = subprocess.run(cmd, capture_output=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found — install it (sudo apt install ffmpeg)")

    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {msg or 'no output'}")

    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def duration_sec(samples, sr: int) -> float:
    return len(samples) / float(sr)
