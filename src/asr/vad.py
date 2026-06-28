"""Silero VAD segmentation: split long audio into speech segments with timestamps.

Models like Qwen3-ASR have a hard ~35 s context limit, so long audio MUST be
chunked. sherpa-onnx's VAD must be fed in small window-sized chunks and drained
incrementally — feeding the whole array at once only emits a final tail segment.
"""

from __future__ import annotations

_WINDOW = 512  # silero VAD window size (samples @16 kHz)


def segment(samples, sr: int, vad_model: str, *, max_segment_sec: float = 20.0,
            min_silence_ms: int = 300, chunk_windows: int = 8):
    """Return list of (start_ms, end_ms, seg_samples) speech segments."""
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_model
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = min_silence_ms / 1000.0
    cfg.silero_vad.max_speech_duration = float(max_segment_sec)
    cfg.silero_vad.min_speech_duration = 0.08
    cfg.sample_rate = sr

    # buffer must exceed the longest single segment (max_seg + silence overrun)
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    step = _WINDOW * chunk_windows
    segments: list[tuple[int, int, object]] = []

    def drain():
        while not vad.empty():
            seg = vad.front
            start = int(getattr(seg, "start", 0))
            audio = getattr(seg, "samples", None)
            if audio is None:
                audio = samples[start:]
            start_ms = int(start * 1000 / sr)
            end_ms = int((start + len(audio)) * 1000 / sr)
            segments.append((start_ms, end_ms, audio))
            vad.pop()

    for i in range(0, len(samples), step):
        piece = samples[i:i + step].tolist()
        vad.accept_waveform(piece)
        drain()

    vad.flush()
    drain()
    return segments
