"""Model downloader: ModelScope + HuggingFace (via mirror), size-verified + resume.

Pure stdlib — no heavy SDK dependency. For each file we learn its expected size
from the source's metadata API, then range-resume the download and verify the
final byte count (ModelScope's CDN omits Content-Length on large files, which is
exactly what silently truncated earlier attempts).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from .models import ModelSpec


def cache_dir() -> str:
    base = (
        os.environ.get("XDG_CACHE_HOME")
        or os.environ.get("ASR_CACHE")
        or (os.path.expanduser("~/.cache"))
    )
    return f"{base}/asr"


def model_dir_for(spec_id: str) -> str:
    return f"{cache_dir()}/{spec_id}"


def _ms_file_sizes(repo: str) -> dict[str, int]:
    """ModelScope: one API call returns every file's size."""
    url = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print(f"  ! cannot query ModelScope metadata: {e}", file=sys.stderr)
        return {}
    out: dict[str, int] = {}
    for f in data.get("Data", {}).get("Files", []):
        path = f.get("Path") or f.get("Name")
        sz = f.get("Size")
        if path and sz:
            try:
                out[path] = int(sz)
            except (TypeError, ValueError):
                pass
    return out


def _hf_file_size(repo: str, file: str) -> int | None:
    base = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    url = f"{base}/{repo}/resolve/main/{file}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _url_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _expected_size(spec: ModelSpec, file: str, ms_sizes: dict[str, int]) -> int | None:
    if spec.source == "modelscope":
        # repo/files API lists root-level files with sizes but not nested dirs;
        # HEAD the resolve URL as a fallback (the CDN sends Content-Length for
        # small files, just not for the multi-hundred-MB ones).
        return ms_sizes.get(file) or _url_size(spec.url_for(file))
    return _hf_file_size(spec.repo, file)


def _fmt(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def _download_file(url: str, dest: Path, expected: int | None, retries: int = 8) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        print(f"  · {dest.name} — present, skipping", file=sys.stderr)
        return
    part = dest.with_suffix(dest.suffix + ".part")
    last_print = 0.0

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resume = have and resp.status == 206
                if not resume:
                    have = 0
                with open(part, "ab" if resume else "wb") as f:
                    while True:
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last_print >= 0.5:
                            last_print = now
                            if expected:
                                pct = min(100.0, have / expected * 100.0)
                                print(f"\r  {dest.name}  {_fmt(have)} / {_fmt(expected)}"
                                      f"  ({pct:.0f}%)   ", end="", file=sys.stderr)
                            else:
                                print(f"\r  {dest.name}  {_fmt(have)}   ",
                                      end="", file=sys.stderr)
            if expected is None or have >= expected:
                if expected and have > expected:
                    part.seek(0, os.SEEK_END)  # trust expected; trim not worth it
                print(file=sys.stderr)
                part.replace(dest)
                return
            print(f"\n  ! {dest.name} short ({have}/{expected}), resuming…",
                  file=sys.stderr)
        except Exception as e:
            print(f"\n  ! {dest.name} attempt {attempt} failed: {e}; resuming…",
                  file=sys.stderr)
            time.sleep(1.5)
    raise RuntimeError(f"failed to download {dest.name} after {retries} attempts")


def download_model(spec: ModelSpec, dest_dir: str | None = None) -> str:
    dest_dir = dest_dir or model_dir_for(spec.id)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    ms_sizes = _ms_file_sizes(spec.repo) if spec.source == "modelscope" else {}
    src_label = "ModelScope" if spec.source == "modelscope" else "HF-mirror"
    print(f"Downloading '{spec.id}' from {src_label} → {dest_dir}",
          file=sys.stderr)

    total = len(spec.files)
    for i, file in enumerate(spec.files, 1):
        print(f"[{i}/{total}] {file}", file=sys.stderr)
        expected = _expected_size(spec, file, ms_sizes)
        url = spec.url_for(file)
        _download_file(url, Path(dest_dir) / file, expected)

    print(f"\n✓ Model '{spec.id}' ready at {dest_dir}", file=sys.stderr)
    return dest_dir


def is_complete(spec: ModelSpec, dest_dir: str) -> bool:
    base = Path(dest_dir)
    return all((base / f).exists() for f in spec.files)


def vad_path() -> str:
    return f"{cache_dir()}/vad/silero_vad.onnx"


def ensure_vad() -> str:
    """Shared silero VAD, for models that don't bundle one. ~2 MB from ModelScope."""
    p = vad_path()
    if os.path.exists(p):
        return p
    from .models import QWEN3_MS
    print("Downloading silero VAD model (~2 MB) ...", file=sys.stderr)
    _download_file(
        f"https://www.modelscope.cn/models/{QWEN3_MS}/resolve/master/silero_vad.onnx",
        Path(p),
        expected=None,
    )
    return p
