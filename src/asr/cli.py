"""asr CLI — local speech-to-text powered by sherpa-onnx."""

from __future__ import annotations

import sys
from typing import Optional

import typer

from . import models
from . import download as dl
from . import engine, output, audio_io

app = typer.Typer(
    name="asr",
    help="Local speech-to-text powered by sherpa-onnx (Qwen3-ASR / SenseVoice / Paraformer / Whisper).",
    no_args_is_help=True,
    add_completion=False,
)


def _err(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    raise typer.Exit(code)


def _status(quiet: bool, msg: str):
    if not quiet:
        print(msg, file=sys.stderr)


@app.command("list")
def list_():
    """List available models."""
    print("Available models:\n")
    for m in models.REGISTRY:
        src = "ModelScope" if m.source == "modelscope" else "HF-mirror"
        print(f"  {m.id:<24} {m.description}")
        print(f"  {'':24} langs={m.languages}  ~{int(m.size_mb)}MB  src={src}"
              + ("  [needs VAD]" if m.needs_vad else ""))
        print()
    print(f"Default: {models.DEFAULT_MODEL}")
    print(f"Cache:   {dl.cache_dir()}")


@app.command()
def download(
    model: Optional[str] = typer.Argument(None, help="model id (default: see `asr default`)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="target directory"),
    set_default: bool = typer.Option(False, "--default", help="set as default after download"),
):
    """Download a model."""
    model = model or models.default_model()
    spec = models.find(model)
    if spec is None:
        print(f"unknown model: '{model}'\n", file=sys.stderr)
        list_()
        raise typer.Exit(1)
    dl.download_model(spec, output)
    if set_default:
        models.set_default_model(spec.id)
        print(f"Default model set to '{spec.id}'", file=sys.stderr)


@app.command()
def default(
    model_id: Optional[str] = typer.Argument(
        None, help="model id to set as default; omit to show the current default"),
):
    """Show or set the default model used when -m/--model is not given."""
    if model_id is None:
        cur = models.default_model()
        spec = models.find(cur)
        print(f"Current default: {cur}")
        if spec:
            print(f"  {spec.description}")
        return
    try:
        models.set_default_model(model_id)
    except ValueError as e:
        _err(str(e))
    print(f"Default model set to '{model_id}'")


@app.command()
def transcribe(
    input_path: str = typer.Argument(..., help="audio/video file, or '-' for stdin"),
    model_id: Optional[str] = typer.Option(None, "--model", "-m", help="registered model id"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="explicit model directory"),
    language: str = typer.Option("", "--language", "-l", help="force language (e.g. zh, en)"),
    hotwords: str = typer.Option("", "--hotwords", help="bias terms (Qwen3-ASR)"),
    fmt: str = typer.Option("text", "--format", "-f", help="text|srt|json"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="write to file"),
    vad: str = typer.Option("auto", "--vad", help="auto|on|off — chunk long audio"),
    max_segment_sec: Optional[float] = typer.Option(None, "--max-segment-sec"),
    threads: int = typer.Option(0, "--threads", "-t", help="0 = all cores"),
    resume: str = typer.Option("auto", "--resume", help="auto|on|off — checkpoint long audio for resume"),
    fresh: bool = typer.Option(False, "--fresh", help="discard any checkpoint and start over"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="transcript only on stdout"),
    auto_download: bool = typer.Option(False, "--auto-download", help="fetch model if missing"),
):
    """Transcribe audio/video to text."""
    if fmt not in ("text", "srt", "json"):
        _err(f"invalid --format '{fmt}' (use text|srt|json)")

    spec_id = model_id or models.default_model()
    spec = models.find(spec_id)
    if spec is None:
        _err(f"unknown model '{spec_id}' (see `asr list`)")

    mdir = model_dir or dl.model_dir_for(spec.id)
    if not dl.is_complete(spec, mdir):
        if auto_download:
            dl.download_model(spec, mdir)
        else:
            _err(f"model '{spec.id}' not found at {mdir}.\n"
                 f"  run: asr download {spec.id}   (or add --auto-download)")

    _status(quiet, f"[model] {spec.id}  [threads] {threads or engine.default_threads()}")
    _status(quiet, f"[audio] loading {input_path} ...")
    try:
        samples, sr = audio_io.load_audio(input_path)
    except Exception as e:
        _err(str(e))

    duration = len(samples) / float(sr)
    _status(quiet, f"[audio] {duration:.1f} s")

    v = vad.lower()
    if v not in ("auto", "on", "off"):
        _err(f"invalid --vad '{vad}' (use auto|on|off)")
    use_vad = None if v == "auto" else (v == "on")

    rv = resume.lower()
    if rv not in ("auto", "on", "off"):
        _err(f"invalid --resume '{resume}' (use auto|on|off)")
    in_path = None if input_path == "-" else input_path

    state = {"cached": 0}
    if not quiet:
        def on_progress(i, total, s_ms, e_ms, cached=False):
            if i >= total:
                return  # completion marker
            if cached:
                state["cached"] += 1
            tag = " (cached)" if cached else ""
            if total > 1:
                print(f"  segment {i + 1}/{total}{tag}  [{s_ms/1000:.1f}s–{e_ms/1000:.1f}s]",
                      file=sys.stderr)
    else:
        on_progress = None

    try:
        segments = engine.transcribe(
            spec, mdir, samples, sr,
            num_threads=threads, language=language, hotwords=hotwords,
            use_vad=use_vad, max_segment_sec=max_segment_sec,
            resume=rv, input_path=in_path, fresh=fresh,
            on_progress=on_progress,
        )
    except Exception as e:
        _err(f"transcription failed: {e}")

    if not quiet and state["cached"]:
        print(f"[resume] reused {state['cached']} cached segment(s)", file=sys.stderr)

    result = {"text": output.to_text, "srt": output.to_srt, "json": output.to_json}[fmt](segments)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        _status(quiet, f"[done] wrote {output_path}")
    else:
        sys.stdout.write(result)
        if not result.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    if not quiet:
        txt = output.to_text(segments)
        nch = len(txt)
        _status(quiet, f"[done] {len(segments)} segment(s), {nch} chars "
                       f"({duration:.1f}s audio)")


def main():
    app()


if __name__ == "__main__":
    main()
