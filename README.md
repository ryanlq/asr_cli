# asr

**本地、纯 CPU、无需联网/API Key 的语音转文字 CLI** —— 以 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)(ONNX Runtime)为引擎,内置多模型:

| 模型 | 语言 | 大小 | 说明 |
|------|------|------|------|
| `qwen3-asr-0.6b-int8` *(默认)* | 多语种 | ~0.9 GB | Qwen3-ASR INT8,综合最强 |
| `sense-voice-small-int8` | zh/en/ja/ko/yue | ~540 MB | SenseVoice,快、带标点 |
| `paraformer-zh-int8` | 中文 | ~220 MB | Paraformer,中文稳 |
| `whisper-tiny-int8` | 多语种 | ~100 MB | 轻量 |
| `whisper-base-int8` | 多语种 | ~155 MB | 比 tiny 准 |

所有模型走 INT8 量化,内存占用低(峰值约 **1.6 GB**)。国内默认从 **ModelScope** 下载(Qwen3/Whisper),SenseVoice/Paraformer 走 **hf-mirror**(HuggingFace 国内镜像)。

## 安装

```bash
cd asr
uv sync                       # 或: pip install -e .
asr --help
```

> 需要 `ffmpeg` 解码非 WAV 音频:`sudo apt install ffmpeg`。
> PyPI 慢时可 `export UV_HTTP_TIMEOUT=600 UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

## 快速开始

```bash
# 1. 下载默认模型(Qwen3-ASR INT8,~0.9GB,ModelScope)
asr download qwen3-asr-0.6b-int8

# 2. 转写(任何音视频格式)
asr transcribe recording.mp3
asr transcribe meeting.wav --format srt -o meeting.srt
asr transcribe call.m4a --language zh --hotwords "K8s,微服务"

# 3. 从标准输入
arecord -q -f S16_LE -r 16000 -c 1 -d 5 | asr transcribe -

# 4. 换模型
asr download whisper-base-int8
asr transcribe clip.wav -m whisper-base-int8 -l en
```

## 命令与选项

```
asr list                          # 列出可选模型
asr default [MODEL]               # 查看 / 切换默认模型(持久化)
asr download <MODEL> [-o DIR] [--default]
asr transcribe <INPUT> [OPTIONS]
```

切换默认模型(之后 `asr transcribe file.wav` 不带 `-m` 就用它):

```bash
asr default                       # 查看当前默认
asr default whisper-base-int8     # 切换默认(写到 ~/.config/asr/default)
asr download sense-voice-small-int8 --default   # 下载并设为默认
```

默认配置存于 `~/.config/asr/default`。

| 选项 | 说明 | 默认 |
|------|------|------|
| `-m, --model <ID>` | 模型 id | `qwen3-asr-0.6b-int8` |
| `--model-dir <DIR>` | 显式模型目录 | `~/.cache/asr/<id>` |
| `-l, --language <LANG>` | 强制语言(`zh`/`en`/`ja`…) | 自动 |
| `--hotwords <TXT>` | 偏置词(Qwen3-ASR) | — |
| `-f, --format <text\|srt\|json>` | 输出格式 | `text` |
| `-o, --output <FILE>` | 写入文件 | stdout |
| `--vad <auto\|on\|off>` | 长音频静音切分 | `auto` |
| `--max-segment-sec <N>` | VAD 分段上限秒数 | 按模型 |
| `-t, --threads <N>` | 线程数 | 全部核 |
| `--resume <auto\|on\|off>` | 长音频断点续传 | `auto` |
| `--fresh` | 丢弃 checkpoint 重来 | 关 |
| `--auto-download` | 缺模型时自动下载 | 关 |
| `-q, --quiet` | 只输出转写文本 | 关 |

## 中断恢复(resume)

长音频(被 VAD 切多段的)默认开启 checkpoint:每识别完一段就 `fsync` 落盘到 `~/.cache/asr/jobs/<哈希>.jsonl`。若中途中断(Ctrl-C、OOM 被杀、崩溃),**重跑同一条命令会自动跳过已完成的段、只补做剩余**,结果与一次跑完逐字节一致;完成后 checkpoint 自动清理。

```bash
asr transcribe 2hour-podcast.mp3 -f srt -o pod.srt   # 跑到一半 Ctrl-C
asr transcribe 2hour-podcast.mp3 -f srt -o pod.srt   # 再跑,自动续上
asr transcribe 2hour-podcast.mp3 --fresh ...          # 丢弃进度,从头来
```

checkpoint 按(音频路径+大小+mtime+模型+VAD参数+语言+热词)哈希定位,文件或参数变了会自动作废重来。短音频(单段)不走 checkpoint。

## 长音频处理

Qwen3-ASR / SenseVoice / Paraformer 有**硬性上下文上限**(约 20–35 秒)。长音频会被 **silero VAD** 自动按静音切分成 ≤20s 的片段(silero 的 soft-limit 之外会**硬性再切**,保证长播客/连续语音不会 OOM),分别识别后拼接;输出 SRT/JSON 时每段带时间戳。(Qwen3-ASR 的模型目录自带 `silero_vad.onnx`;其它模型首次用 VAD 会自动拉取一个共享的 VAD,~2MB。)

## 项目结构

```
src/asr/
  models.py    模型注册表 + 各 family 的 recognizer 构造
  download.py  ModelScope / HF-mirror 下载(带大小校验、断点续传)
  audio_io.py  ffmpeg 解码任意格式 → 16k 单声道 f32
  vad.py       silero VAD 分段
  engine.py    加载引擎 + 转写(按需 VAD 切分)
  output.py    text / srt / json 格式化
  cli.py       typer CLI
```

## License

MIT
