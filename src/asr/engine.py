"""Build a recognizer and run transcription, with VAD chunking for long audio."""

from __future__ import annotations

import os

from . import models, vad as vad_mod


def default_threads() -> int:
    return os.cpu_count() or 4


def transcribe(spec: "models.ModelSpec", model_dir: str, samples, sr: int, *,
               num_threads: int = 0, language: str = "", hotwords: str = "",
               use_vad: bool | None = None, max_segment_sec: float | None = None,
               resume: str = "auto", input_path: str | None = None, fresh: bool = False,
               on_progress=None):
    """Return list of (start_ms, end_ms, text) segments.

    When ``resume`` is auto/on and the audio is VAD-chunked, each completed
    segment is checkpointed to disk, so an interrupted run (Ctrl-C, OOM kill,
    crash) continues from where it stopped on the next invocation.
    """
    nt = num_threads or default_threads()
    rec = models.build_recognizer(spec, model_dir, num_threads=nt,
                                  language=language, hotwords=hotwords)

    duration = len(samples) / float(sr)
    seg_cap = max_segment_sec or spec.max_segment_sec
    if use_vad is None:
        use_vad = spec.needs_vad and duration > seg_cap

    def single():
        if on_progress:
            on_progress(0, 1, 0, int(duration * 1000))
        text = _decode(rec, samples, sr)
        if on_progress:
            on_progress(1, 1, 0, 0)
        return [(0, int(duration * 1000), text)]

    if not use_vad:
        return single()

    vad_model = _resolve_vad(spec, model_dir)
    segs = vad_mod.segment(samples, sr, vad_model, max_segment_sec=seg_cap)
    if not segs:
        return single()

    n = len(segs)
    results: list[tuple[int, int, str] | None] = [None] * n

    # --- checkpoint / resume ---
    writer = None
    ckpt = None
    if resume in ("auto", "on") and input_path:
        from . import checkpoint as ck
        try:
            st = os.stat(input_path)
            jid = ck.job_id(input_path, st.st_size, int(st.st_mtime),
                            spec.id, seg_cap, language, hotwords)
            cp = ck.ckpt_path(jid)
            done = None if fresh else ck.load(cp, n)
            if done is None:
                ck.discard(cp)
            else:
                for i, seg in done.items():
                    if 0 <= i < n:
                        results[i] = seg
            writer = ck.Writer(cp, n)
            ckpt = cp
        except OSError:
            writer = None  # filesystem error → run without checkpointing

    for i, (s_ms, e_ms, audio) in enumerate(segs):
        cached = results[i] is not None
        if on_progress:
            on_progress(i, n, s_ms, e_ms, cached=cached)
        if cached:
            continue
        text = _decode(rec, audio, sr)
        results[i] = (s_ms, e_ms, text)
        if writer:
            writer.save(i, s_ms, e_ms, text)

    if writer:
        writer.close()
    if on_progress:
        on_progress(n, n, segs[-1][1], segs[-1][1])  # completion marker

    if ckpt is not None:
        from . import checkpoint as ck
        ck.discard(ckpt)  # job finished → drop the checkpoint

    return [r for r in results if r is not None]


def _decode(rec, samples, sr: int) -> str:
    stream = rec.create_stream()
    stream.accept_waveform(sr, samples)
    rec.decode_stream(stream)
    return stream.result.text.strip()


def _resolve_vad(spec: "models.ModelSpec", model_dir: str) -> str:
    bundled = os.path.join(model_dir, "silero_vad.onnx")
    if os.path.exists(bundled):
        return bundled
    from .download import ensure_vad
    return ensure_vad()
