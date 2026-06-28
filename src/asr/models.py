"""Model registry: metadata + per-family recognizer builders.

Each entry knows which sherpa-onnx loader to use, which files to download, and
where to fetch them (ModelScope for speed in CN, HuggingFace via mirror as
fallback). Adding a new model = one more ``ModelSpec``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# A model source: ("modelscope", repo_id) or ("huggingface", repo_id).
QWEN3_MS = "jkman2023/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
WHISPER_TINY_MS = "pengzhendong/sherpa-onnx-whisper-tiny"
WHISPER_BASE_MS = "pengzhendong/sherpa-onnx-whisper-base"
SENSEVOICE_HF = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
PARAFORMER_HF = "csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14"


@dataclass
class ModelSpec:
    id: str                       # unique registry id
    family: str                   # qwen3_asr | sense_voice | paraformer | whisper
    source: str                   # "modelscope" | "huggingface"
    repo: str                     # repo id on that source
    files: list[str]              # files to fetch (repo-relative; may include dirs)
    description: str
    size_mb: float
    languages: str = ""           # e.g. "zh,en,ja,ko,yue" or "multilingual"
    max_segment_sec: float = 25.0 # suggested VAD ceiling for this model
    needs_vad: bool = False       # has a hard context limit → VAD recommended
    extra: dict = field(default_factory=dict)

    def url_for(self, file: str) -> str:
        if self.source == "modelscope":
            return f"https://www.modelscope.cn/models/{self.repo}/resolve/master/{file}"
        base = _hf_base()
        return f"{base}/{self.repo}/resolve/main/{file}"


def _hf_base() -> str:
    import os
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


REGISTRY: list[ModelSpec] = [
    ModelSpec(
        id="qwen3-asr-0.6b-int8",
        family="qwen3_asr",
        source="modelscope",
        repo=QWEN3_MS,
        files=[
            "conv_frontend.onnx",
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "silero_vad.onnx",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
        ],
        description="Qwen3-ASR 0.6B (INT8) — best multilingual accuracy, ~0.9 GB",
        size_mb=940,
        languages="zh,en,ja,ko,yue,ar,de,es,fr,ru,...",
        max_segment_sec=20,
        needs_vad=True,
    ),
    ModelSpec(
        id="sense-voice-small-int8",
        family="sense_voice",
        source="huggingface",
        repo=SENSEVOICE_HF,
        files=["model.int8.onnx", "tokens.txt"],
        description="SenseVoice Small (INT8) — fast zh/en/ja/ko/yue, ~540 MB",
        size_mb=540,
        languages="zh,en,ja,ko,yue",
        max_segment_sec=30,
        needs_vad=True,
    ),
    ModelSpec(
        id="paraformer-zh-int8",
        family="paraformer",
        source="huggingface",
        repo=PARAFORMER_HF,
        files=["model.int8.onnx", "tokens.txt"],
        description="Paraformer (INT8) — solid Chinese workhorse, ~220 MB",
        size_mb=220,
        languages="zh",
        max_segment_sec=20,
        needs_vad=True,
    ),
    ModelSpec(
        id="whisper-tiny-int8",
        family="whisper",
        source="modelscope",
        repo=WHISPER_TINY_MS,
        files=["tiny-encoder.int8.onnx", "tiny-decoder.int8.onnx", "tiny-tokens.txt"],
        description="Whisper tiny (INT8) — lightweight multilingual, ~100 MB",
        size_mb=100,
        languages="multilingual",
        max_segment_sec=30,
        extra={"prefix": "tiny"},
    ),
    ModelSpec(
        id="whisper-base-int8",
        family="whisper",
        source="modelscope",
        repo=WHISPER_BASE_MS,
        files=["base-encoder.int8.onnx", "base-decoder.int8.onnx", "base-tokens.txt"],
        description="Whisper base (INT8) — better multilingual, ~155 MB",
        size_mb=155,
        languages="multilingual",
        max_segment_sec=30,
        extra={"prefix": "base"},
    ),
]

DEFAULT_MODEL = "qwen3-asr-0.6b-int8"


def find(spec_id: str) -> ModelSpec | None:
    for m in REGISTRY:
        if m.id == spec_id:
            return m
    return None


def config_dir() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return f"{base}/asr"


def _default_file() -> Path:
    return Path(config_dir()) / "default"


def default_model() -> str:
    """The effective default model id (persisted config, else DEFAULT_MODEL)."""
    try:
        f = _default_file()
        if f.exists():
            val = f.read_text(encoding="utf-8").strip()
            if val and find(val) is not None:
                return val
    except OSError:
        pass
    return DEFAULT_MODEL


def set_default_model(spec_id: str) -> None:
    """Persist the default model id (must be a registered id)."""
    if find(spec_id) is None:
        raise ValueError(f"unknown model '{spec_id}' (see `asr list`)")
    f = _default_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(spec_id, encoding="utf-8")


def build_recognizer(spec: ModelSpec, model_dir: str, *, num_threads: int,
                     language: str = "", hotwords: str = ""):
    """Lazy-import sherpa_onnx and build the right recognizer for this family."""
    import sherpa_onnx

    d = model_dir
    if spec.family == "qwen3_asr":
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=f"{d}/conv_frontend.onnx",
            encoder=f"{d}/encoder.int8.onnx",
            decoder=f"{d}/decoder.int8.onnx",
            tokenizer=f"{d}/tokenizer",
            num_threads=num_threads,
            max_total_len=512,
            max_new_tokens=256,
            hotwords=hotwords,
        )
    if spec.family == "sense_voice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{d}/model.int8.onnx",
            tokens=f"{d}/tokens.txt",
            num_threads=num_threads,
            language=language or "auto",
            use_itn=True,
        )
    if spec.family == "paraformer":
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=f"{d}/model.int8.onnx",
            tokens=f"{d}/tokens.txt",
            num_threads=num_threads,
        )
    if spec.family == "whisper":
        p = spec.extra.get("prefix", "tiny")
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=f"{d}/{p}-encoder.int8.onnx",
            decoder=f"{d}/{p}-decoder.int8.onnx",
            tokens=f"{d}/{p}-tokens.txt",
            language=language or "en",
            num_threads=num_threads,
        )
    raise ValueError(f"unknown family: {spec.family}")
