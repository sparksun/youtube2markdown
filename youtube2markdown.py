#!/usr/bin/env python3
"""
youtube2markdown.py
────────────────────────────────────────────────────────────────
将 YouTube 视频或本地媒体文件（MP4 等）转成通顺的中文 Markdown 文档。

流程（YouTube URL）：
  1. 用 youtube-transcript-api 获取字幕（有则优先）
  2. 无字幕时，用 yt-dlp 下载音频 + Whisper 转录（可选依赖）
  3. 通过 OpenRouter → deepseek/deepseek-v4-flash 纠错/翻译/重写
  4. 输出带 YAML frontmatter 的 Markdown 文件到 output/ 目录

流程（本地文件）：
  1. 用 ffmpeg 从 MP4（或其他媒体文件）中提取音频
  2. Whisper 转录（含实时进度显示）
  3. 通过 OpenRouter → deepseek/deepseek-v4-flash 纠错/翻译/重写
  4. 输出带 YAML frontmatter 的 Markdown 文件到 output/ 目录

转录后端（--whisper-backend）：
  auto           自动选择：Apple Silicon → mlx，其他 → faster-whisper
  mlx            mlx-whisper（Apple Silicon GPU/ANE，5–10× 加速）
  faster-whisper 传统 CPU 推理（beam_size=1 + VAD 已优化）

用法：
  python youtube2markdown.py <YouTube URL> [选项]
  python youtube2markdown.py <本地文件路径> [选项]

选项：
  -o, --output <文件名>        指定输出文件名（默认：根据视频标题/文件名自动生成）
  --output-dir <目录>          指定输出目录（默认：./output/）
  --force-whisper              强制使用 Whisper（忽略已有字幕，仅 YouTube 模式）
  --whisper-model <size>       Whisper 模型大小：tiny/base/small/medium/large（默认：medium）
  --whisper-backend <backend>  转录后端：auto/mlx/faster-whisper（默认：auto）
  --chunk-size <字符数>        DeepSeek 每次处理的字符数（默认：30000，DeepSeek 上下文 1M）
  --lang <语言代码>            手动指定源语言（如 en/ja/zh），本地文件模式下 Whisper 自动检测
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from openai import OpenAI

# ── 加载环境变量 ────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── OpenRouter 客户端 ───────────────────────────────────────────
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "deepseek/deepseek-v4-flash"

# 支持直接处理的本地媒体扩展名
LOCAL_MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",          # 视频
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg",  # 音频
}

# mlx-whisper 模型仓库映射
MLX_MODEL_MAP = {
    "tiny":   "mlx-community/whisper-tiny-mlx",
    "base":   "mlx-community/whisper-base-mlx",
    "small":  "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large":  "mlx-community/whisper-large-v3-mlx",
}


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def is_local_file(path_or_url: str) -> bool:
    """判断输入是本地文件路径还是 URL。"""
    p = Path(path_or_url)
    if p.exists() and p.is_file():
        return True
    # 如果文件不存在但扩展名是媒体格式，也视为本地文件（报错交给后续处理）
    if p.suffix.lower() in LOCAL_MEDIA_EXTENSIONS and not path_or_url.startswith("http"):
        return True
    return False


def extract_video_id(url: str) -> str:
    """从 YouTube URL 中提取 video_id。"""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        parts = parsed.path.split("/")
        if "shorts" in parts:
            idx = parts.index("shorts")
            return parts[idx + 1]
    raise ValueError(f"无法从 URL 中提取 video_id：{url}")


def safe_filename(title: str, max_len: int = 80) -> str:
    """将视频标题转成安全的文件名。"""
    name = re.sub(r'[\\/*?"<>|]', "_", title)
    name = name.strip().strip(".")
    return name[:max_len] if len(name) > max_len else name


def chunk_text(text: str, chunk_size: int = 30000) -> list[str]:
    """将长文本按句子边界分段，每段不超过 chunk_size 字符。"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                sentences = re.split(r"([。！？\.!?]\s*)", para)
                current = ""
                for i in range(0, len(sentences), 2):
                    sent = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                    if len(current) + len(sent) <= chunk_size:
                        current += sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


# ═══════════════════════════════════════════════════════════════
# 转录后端选择
# ═══════════════════════════════════════════════════════════════

def _has_cuda() -> bool:
    """
    检测是否有可用的 NVIDIA CUDA GPU。
    依次尝试四种方法，兼容 ctranslate2 3.x / 4.x 以及有无 torch。
    """
    # 方法 1：ctranslate2 4.x 提供的 GPU 计数接口（最可靠）
    try:
        import ctranslate2
        if hasattr(ctranslate2, "get_cuda_device_count"):
            if ctranslate2.get_cuda_device_count() > 0:
                return True
    except Exception:
        pass

    # 方法 2：ctranslate2 支持的计算类型列表（适用于部分 3.x/4.x）
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        if types:
            return True
    except Exception:
        pass

    # 方法 3：nvidia-smi（不依赖 Python 包，只要驱动已装就能用）
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass

    # 方法 4：torch（如果已安装）
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    return False


def _ctranslate2_cuda_compiled() -> bool:
    """
    检测当前安装的 ctranslate2 是否编译了 CUDA 支持。
    ctranslate2 在 ARM64 平台（DGX Spark 等）无官方 CUDA wheel，当返回 False 时
    应考虑使用 openai-whisper + torch 作为替代方案。
    """
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        return bool(types)
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    """torch 已安装且 CUDA 可用，失败返回 False。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_best_backend() -> str:
    """
    自动检测最佳转录后端。
    优先级：
      1. Apple Silicon (arm64 Mac) + mlx-whisper 已安装 → mlx（GPU/ANE，5–10×）
      2. CUDA GPU 可用 + ctranslate2 已编译 CUDA → faster-whisper（GPU float16，~10–20×）
      3. CUDA GPU 可用 + ctranslate2 无 CUDA + torch 已安装 → torch（GPU，适用 ARM64）
      4. 其他 → faster-whisper（CPU int8）
    """
    # 1. Apple Silicon + mlx
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            print("  ℹ️  未安装 mlx-whisper（pip install mlx-whisper），回退到 faster-whisper")

    # 2 & 3. 检测 CUDA
    if _has_cuda():
        if _ctranslate2_cuda_compiled():
            # ctranslate2 已编译 CUDA，直接用 faster-whisper GPU
            return "faster-whisper"
        # ctranslate2 未编译 CUDA（常见于 ARM64 如 DGX Spark）
        if _torch_cuda_available():
            try:
                import whisper  # noqa: F401
                print("  ℹ️  ctranslate2 无 CUDA 支持（ARM64 平台），改用 openai-whisper + torch GPU")
                return "torch"
            except ImportError:
                print("  ℹ️  ctranslate2 无 CUDA 支持，建议：pip install openai-whisper torch")
        else:
            print("  ℹ️  ctranslate2 无 CUDA 支持，建议安装 torch 以启用 GPU：pip install openai-whisper torch")

    return "faster-whisper"


# ═══════════════════════════════════════════════════════════════
# 转录后端 A：faster-whisper（已调优）
# ═══════════════════════════════════════════════════════════════

def _transcribe_faster_whisper(
    audio_path: str,
    model_size: str = "medium",
) -> tuple[str, str]:
    """
    用 faster-whisper 转录音频，实时显示进度条。
    自动选择设备：NVIDIA GPU（float16）> CPU（int8）。
    优化参数：beam_size=1（贪心解码）+ vad_filter（跳过静音段）。
    返回 (transcript_text, detected_language)
    """
    from faster_whisper import WhisperModel

    print(f"🎙️  [faster-whisper / {model_size}] 检测设备……")

    # ── 自动选择设备：先尝试 CUDA，失败则回退 CPU ──────────────────────
    if _has_cuda():
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            device_label = "NVIDIA GPU / float16"
        except Exception as cuda_err:
            print(f"  ⚠️  CUDA 初始化失败（{cuda_err}），回退到 CPU")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            device_label = "CPU / int8"
    else:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        device_label = "CPU / int8"

    print(f"🎙️  [faster-whisper / {model_size} / {device_label}] 转录中（beam_size=1, VAD 已启用）……")
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,        # 贪心解码，2–3× 加速，精度损失极小
        vad_filter=True,    # 跳过静音片段，减少 10–30% 处理量
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    audio_duration = info.duration
    text_parts = []
    seg_count = 0
    start_time = time.time()
    last_print_pos = -1.0

    for seg in segments:
        text_parts.append(seg.text)
        seg_count += 1
        current_pos = seg.end

        if current_pos - last_print_pos >= 30 or seg_count == 1:
            elapsed = time.time() - start_time
            if audio_duration and audio_duration > 0:
                pct = min(current_pos / audio_duration * 100, 100)
                if pct > 0:
                    eta = elapsed / (pct / 100) * (1 - pct / 100)
                    eta_str = f"  预计剩余 {eta:.0f}s"
                else:
                    eta_str = ""
                bar_filled = int(pct / 5)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                print(
                    f"  [{bar}] {pct:5.1f}%  "
                    f"{int(current_pos)//60:02d}:{int(current_pos)%60:02d} / "
                    f"{int(audio_duration)//60:02d}:{int(audio_duration)%60:02d}  "
                    f"已用 {elapsed:.0f}s{eta_str}"
                )
            else:
                print(f"  已转录 {seg_count} 段  已用时 {elapsed:.0f}s")
            last_print_pos = current_pos

    elapsed_total = time.time() - start_time
    print(f"✅ 转录完成，共 {seg_count} 段，耗时 {elapsed_total:.1f}s")
    return " ".join(text_parts), info.language


# ═══════════════════════════════════════════════════════════════
# 转录后端 B：mlx-whisper（Apple Silicon GPU/ANE）
# ═══════════════════════════════════════════════════════════════

def _transcribe_mlx(
    audio_path: str,
    model_size: str = "medium",
) -> tuple[str, str]:
    """
    用 mlx-whisper 转录音频（Apple Silicon GPU/ANE 加速，5–10× 快于 CPU）。
    mlx-whisper 不支持逐段迭代，用 spinner + 计时显示进度。
    返回 (transcript_text, detected_language)
    """
    try:
        import mlx_whisper
    except ImportError:
        raise ImportError(
            "mlx-whisper 未安装。\n"
            "请运行：pip install mlx-whisper\n"
            "（仅 Apple Silicon Mac 支持）"
        )

    repo = MLX_MODEL_MAP.get(model_size, MLX_MODEL_MAP["medium"])
    print(f"🚀 [mlx-whisper / {model_size}] Apple Silicon 加速转录……")
    print(f"   模型仓库：{repo}（首次运行会自动下载）")

    # mlx_whisper.transcribe 是阻塞调用，放入子线程以便主线程显示进度
    result_holder: list = [None]
    error_holder:  list = [None]

    def _run():
        try:
            result_holder[0] = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=repo,
            )
        except Exception as exc:
            error_holder[0] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    # Spinner + 计时（主线程）
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()
    idx = 0
    while worker.is_alive():
        elapsed = time.time() - start_time
        print(
            f"\r  {spinner[idx % len(spinner)]}  转录中……  已用 {elapsed:.0f}s",
            end="",
            flush=True,
        )
        time.sleep(0.15)
        idx += 1
    print()  # 换行，清除 spinner

    worker.join()

    if error_holder[0]:
        raise error_holder[0]

    elapsed_total = time.time() - start_time
    result = result_holder[0]

    # mlx_whisper 返回 dict，key: "text", "language", "segments" …
    text = result.get("text", "").strip()
    language = result.get("language", "unknown")

    print(f"✅ 转录完成，耗时 {elapsed_total:.1f}s")
    return text, language


# ═══════════════════════════════════════════════════════════════
# 转录后端 C：openai-whisper + torch（ARM64 CUDA 等层）
# ═══════════════════════════════════════════════════════════════

def _transcribe_torch_whisper(
    audio_path: str,
    model_size: str = "medium",
) -> tuple[str, str]:
    """
    用 openai-whisper + PyTorch 转录音频。
    适用于 ctranslate2 无 CUDA 支持的平台（如 ARM64 DGX Spark）。
    PyTorch 对 ARM64+CUDA 有官方支持，可直接使用 GPU。
    转录为阻塞调用，用 threading + spinner 显示进度。
    返回 (transcript_text, detected_language)
    """
    try:
        import whisper
        import torch
    except ImportError as e:
        raise ImportError(
            f"缺少依赖：{e}\n"
            "请运行：pip install openai-whisper torch"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = "NVIDIA GPU" if device == "cuda" else "CPU"
    print(f"🚀 [openai-whisper / {model_size} / {device_label}] 加载模型……")
    model = whisper.load_model(model_size, device=device)
    print(f"🎙️  开始转录（首次运行不需下载）……")

    result_holder: list = [None]
    error_holder:  list = [None]

    def _run():
        try:
            # verbose=False 关闭 openai-whisper 自带的进度输出
            result_holder[0] = model.transcribe(audio_path, verbose=False)
        except Exception as exc:
            error_holder[0] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()
    idx = 0
    while worker.is_alive():
        elapsed = time.time() - start_time
        print(
            f"\r  {spinner[idx % len(spinner)]}  转录中……  已用 {elapsed:.0f}s",
            end="",
            flush=True,
        )
        time.sleep(0.15)
        idx += 1
    print()  # 换行清除 spinner

    worker.join()
    if error_holder[0]:
        raise error_holder[0]

    elapsed_total = time.time() - start_time
    result = result_holder[0]
    text = result.get("text", "").strip()
    language = result.get("language", "unknown")
    print(f"✅ 转录完成，耗时 {elapsed_total:.1f}s")
    return text, language


# ═══════════════════════════════════════════════════════════════
# 核心：转录调度器
# ═══════════════════════════════════════════════════════════════

def _transcribe_audio_file(
    audio_path: str,
    model_size: str = "medium",
    backend: str = "auto",
) -> tuple[str, str]:
    """
    选择最佳转录后端并执行转录。
    backend: "auto" | "mlx" | "faster-whisper" | "torch"
    返回 (transcript_text, detected_language)
    """
    resolved = backend
    if backend == "auto":
        resolved = _detect_best_backend()
        print(f"  🔍 自动选择转录后端：{resolved}")

    if resolved == "mlx":
        return _transcribe_mlx(audio_path, model_size)
    elif resolved == "torch":
        return _transcribe_torch_whisper(audio_path, model_size)
    else:
        return _transcribe_faster_whisper(audio_path, model_size)


# ═══════════════════════════════════════════════════════════════
# Step 1-A：YouTube —— 获取视频元数据 + 字幕
# ═══════════════════════════════════════════════════════════════

def get_video_info_yt_dlp(url: str) -> dict:
    """用 yt-dlp 获取视频元数据（不下载视频）。"""
    try:
        import yt_dlp
    except ImportError:
        return {}

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title", "未知标题"),
        "channel": info.get("uploader", info.get("channel", "未知频道")),
        "upload_date": info.get("upload_date", ""),
        "description": info.get("description", ""),
        "duration": info.get("duration", 0),
    }


def get_transcript_api(video_id: str) -> tuple[str, str]:
    """
    用 youtube-transcript-api 获取字幕。
    返回 (transcript_text, detected_language)
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    tl = api.list_transcripts(video_id)

    preferred = ["zh", "zh-Hans", "zh-TW", "zh-Hant", "en", "ja"]
    transcript = None
    lang_code = None

    for code in preferred:
        try:
            transcript = tl.find_transcript([code])
            lang_code = code
            break
        except Exception:
            continue

    if transcript is None:
        transcript = next(iter(tl))
        lang_code = transcript.language_code

    entries = transcript.fetch()
    text_parts = []
    for e in entries:
        if isinstance(e, dict):
            text_parts.append(e.get("text", ""))
        else:
            text_parts.append(getattr(e, "text", str(e)))
    return " ".join(text_parts), lang_code


def download_audio_and_transcribe(
    url: str,
    whisper_model_size: str = "medium",
    backend: str = "auto",
) -> tuple[str, str]:
    """
    用 yt-dlp 下载音频，再转录。
    返回 (transcript_text, detected_language)
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError(
            f"需要安装 yt-dlp：{e}\n请运行：pip install yt-dlp"
        ) from e

    print("⬇️  无字幕，正在下载音频（可能需要数分钟）……")

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        downloaded = list(Path(tmpdir).glob("audio.*"))
        if not downloaded:
            raise FileNotFoundError("音频下载失败")
        audio_path = str(downloaded[0])

        return _transcribe_audio_file(audio_path, whisper_model_size, backend)


# ═══════════════════════════════════════════════════════════════
# Step 1-B：本地文件 —— 提取音频并转录
# ═══════════════════════════════════════════════════════════════

def _check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用。"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def transcribe_local_file(
    file_path: str,
    whisper_model_size: str = "medium",
    backend: str = "auto",
) -> tuple[str, str]:
    """
    从本地媒体文件提取音频，再转录。
    返回 (transcript_text, detected_language)
    """
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    suffix = src.suffix.lower()
    audio_only_formats = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg"}

    if suffix in audio_only_formats:
        print(f"🎵 检测到音频文件，直接转录：{src.name}")
        return _transcribe_audio_file(str(src), whisper_model_size, backend)

    # 视频格式：先用 ffmpeg 提取音频
    if not _check_ffmpeg():
        raise RuntimeError(
            "未找到 ffmpeg，无法从视频中提取音频。\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg"
        )

    print(f"🎬 本地文件：{src.name}（{src.stat().st_size / 1024 / 1024:.1f} MB）")
    print("🔧 正在用 ffmpeg 提取音频……")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        cmd = [
            "ffmpeg", "-i", str(src),
            "-vn", "-ar", "16000", "-ac", "1", "-b:a", "128k",
            "-f", "mp3", audio_path,
            "-y", "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 提取音频失败：\n{result.stderr}")
        if not Path(audio_path).exists():
            raise FileNotFoundError("ffmpeg 提取音频后文件不存在")

        audio_size_mb = Path(audio_path).stat().st_size / 1024 / 1024
        print(f"✅ 音频提取完成（{audio_size_mb:.1f} MB），开始转录……")

        return _transcribe_audio_file(audio_path, whisper_model_size, backend)


# ═══════════════════════════════════════════════════════════════
# Step 2：DeepSeek 重写
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一位专业的内容整理专家，擅长将视频字幕/语音转录文字整理成高质量的中文文章。
你的任务是将用户提供的文本（可能是英文、日文或中文的视频字幕，或语音识别结果）整理成通顺、专业的中文内容。

整理要求：
1. 如果是非中文内容，翻译成中文
2. 纠正语音识别或字幕中的错误（词汇混淆、断句错误等）
3. 去除口语化的语气词、重复表达（如"嗯"、"那个"、"然后然后"等）
4. 合并相关内容，整理成有逻辑的段落
5. 保持原文的核心信息和观点，不添加主观意见
6. 专业术语保持准确（如 AI、API、SaaS 等可保留英文）
7. 输出纯中文正文，不需要加标题"""


def call_deepseek(text: str, source_lang: str = "en", is_asr: bool = False) -> str:
    """调用 OpenRouter → DeepSeek 重写单个文本片段。"""
    lang_desc = {
        "zh": "中文", "zh-Hans": "中文", "zh-TW": "中文", "zh-Hant": "中文",
        "en": "英文", "ja": "日文",
    }.get(source_lang, source_lang)

    extra = "（注意：这是语音识别结果，可能含有较多识别错误，请重点纠错）" if is_asr else ""
    user_prompt = f"以下是来自视频的{lang_desc}字幕内容{extra}，请整理为通顺的中文段落：\n\n{text}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_metadata(title: str, full_content: str) -> dict:
    """让 DeepSeek 生成摘要和标签。"""
    prompt = f"""视频标题：{title}

以下是视频的整理内容（摘录）：
{full_content[:2000]}

请用 JSON 格式返回：
{{
  "summary": "3-5句话的中文摘要",
  "tags": ["标签1", "标签2", "标签3"]
}}
只返回 JSON，不要其他文字。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        import json
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"summary": "", "tags": []}


def rewrite_to_chinese(
    raw_text: str,
    source_lang: str,
    is_asr: bool = False,
    chunk_size: int = 30000,
) -> str:
    """将原始字幕文本分段重写为中文，返回完整重写后的正文。"""
    chunks = chunk_text(raw_text, chunk_size)
    total = len(chunks)
    results = []

    for i, chunk in enumerate(chunks, 1):
        print(f"🤖 DeepSeek 处理中 [{i}/{total}]……")
        rewritten = call_deepseek(chunk, source_lang=source_lang, is_asr=is_asr)
        results.append(rewritten)

    return "\n\n".join(results)


# ═══════════════════════════════════════════════════════════════
# Step 3：生成 Markdown
# ═══════════════════════════════════════════════════════════════

def build_markdown(
    source_ref: str,
    video_info: dict,
    source_lang: str,
    summary: str,
    tags: list[str],
    content: str,
    is_local: bool = False,
) -> str:
    """组装最终的 Markdown 文档。"""
    title = video_info.get("title", "未知标题")
    channel = video_info.get("channel", "")
    upload_date = video_info.get("upload_date", "")

    if upload_date and len(upload_date) == 8:
        date_str = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    generated_date = datetime.now().strftime("%Y-%m-%d")

    lang_display = {
        "zh": "中文", "zh-Hans": "中文（简体）", "zh-TW": "中文（繁体）",
        "zh-Hant": "中文（繁体）", "en": "英文", "ja": "日文",
    }.get(source_lang, source_lang)

    tags_yaml = "\n".join(f'  - "{t}"' for t in tags)

    if is_local:
        source_field = f'file: "{Path(source_ref).name}"'
    else:
        source_field = f'source: "{source_ref}"'

    channel_line = f'channel: "{channel}"\n' if channel else ""

    frontmatter = f"""---
title: "{title}"
{source_field}
{channel_line}date: "{date_str}"
language: "{source_lang}"
tags:
{tags_yaml}
generated: "{generated_date}"
tool: "youtube2markdown"
---"""

    summary_block = f"> **摘要**：{summary}\n" if summary else ""

    if is_local:
        footer = (
            f"*本文由 [youtube2markdown](https://github.com/) 自动生成 · "
            f"来源：本地文件 · 原始语言：{lang_display} · 生成日期：{generated_date}*"
        )
    else:
        footer = (
            f"*本文由 [youtube2markdown](https://github.com/) 自动生成 · "
            f"原视频语言：{lang_display} · 生成日期：{generated_date}*"
        )

    doc = f"""{frontmatter}

# {title}

{summary_block}
---

{content}

---

{footer}
"""
    return doc


# ═══════════════════════════════════════════════════════════════
# 主流程 A：YouTube URL
# ═══════════════════════════════════════════════════════════════

def process(
    url: str,
    output_dir: str = "output",
    output_filename: str | None = None,
    force_whisper: bool = False,
    whisper_model: str = "medium",
    whisper_backend: str = "auto",
    chunk_size: int = 30000,
) -> str:
    """完整处理流程：YouTube URL → Markdown 文件。返回输出文件路径。"""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "未找到 OPENROUTER_API_KEY。\n"
            "请在 .env 文件中设置：OPENROUTER_API_KEY=sk-or-v1-..."
        )

    print(f"\n{'═' * 60}")
    print(f"📺 YouTube → Markdown 转换工具")
    print(f"{'═' * 60}")
    print(f"🔗 URL：{url}\n")

    video_id = extract_video_id(url)
    print(f"🎬 Video ID：{video_id}")

    print("📋 获取视频信息……")
    try:
        video_info = get_video_info_yt_dlp(url)
    except Exception as e:
        print(f"⚠️  无法获取完整视频信息（{e}），将使用基础信息。")
        video_info = {}

    if not video_info:
        video_info = {"title": f"YouTube_{video_id}", "channel": "", "upload_date": ""}

    print(f"📌 标题：{video_info.get('title', 'N/A')}")
    print(f"📡 频道：{video_info.get('channel', 'N/A')}")

    raw_text = ""
    source_lang = "en"
    is_asr = False

    if not force_whisper:
        try:
            print("📝 尝试获取 YouTube 字幕……")
            raw_text, source_lang = get_transcript_api(video_id)
            print(f"✅ 字幕语言：{source_lang}，共 {len(raw_text)} 字符")
        except Exception as e:
            print(f"⚠️  字幕获取失败（{e}），回退到 Whisper ASR……")

    if not raw_text:
        raw_text, source_lang = download_audio_and_transcribe(url, whisper_model, whisper_backend)
        is_asr = True
        print(f"✅ Whisper 识别语言：{source_lang}，共 {len(raw_text)} 字符")

    print(f"\n✍️  开始 DeepSeek 重写（模型：{MODEL}）……")
    content = rewrite_to_chinese(raw_text, source_lang, is_asr=is_asr, chunk_size=chunk_size)

    print("🏷️  生成摘要和标签……")
    try:
        meta = generate_metadata(video_info.get("title", ""), content)
        summary = meta.get("summary", "")
        tags = meta.get("tags", [])
    except Exception as e:
        print(f"⚠️  元数据生成失败（{e}）")
        summary = ""
        tags = []

    markdown = build_markdown(url, video_info, source_lang, summary, tags, content, is_local=False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename:
        out_path = out_dir / output_filename
        if not out_path.suffix:
            out_path = out_path.with_suffix(".md")
    else:
        fname = safe_filename(video_info.get("title", f"YouTube_{video_id}"))
        out_path = out_dir / f"{fname}.md"

    out_path.write_text(markdown, encoding="utf-8")

    print(f"\n{'═' * 60}")
    print(f"✅ 完成！输出文件：{out_path.resolve()}")
    print(f"{'═' * 60}\n")

    return str(out_path.resolve())


# ═══════════════════════════════════════════════════════════════
# 主流程 B：本地媒体文件
# ═══════════════════════════════════════════════════════════════

def process_local(
    file_path: str,
    output_dir: str = "output",
    output_filename: str | None = None,
    whisper_model: str = "medium",
    whisper_backend: str = "auto",
    chunk_size: int = 30000,
    lang_hint: str | None = None,
) -> str:
    """完整处理流程：本地媒体文件 → Markdown 文件。返回输出文件路径。"""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "未找到 OPENROUTER_API_KEY。\n"
            "请在 .env 文件中设置：OPENROUTER_API_KEY=sk-or-v1-..."
        )

    src = Path(file_path).resolve()

    print(f"\n{'═' * 60}")
    print(f"🎬 本地文件 → Markdown 转换工具")
    print(f"{'═' * 60}")
    print(f"📂 文件：{src}\n")

    raw_text, detected_lang = transcribe_local_file(str(src), whisper_model, whisper_backend)
    source_lang = lang_hint if lang_hint else detected_lang
    print(f"✅ Whisper 识别语言：{detected_lang}，共 {len(raw_text)} 字符")
    if lang_hint and lang_hint != detected_lang:
        print(f"ℹ️  已按手动指定语言 {lang_hint} 处理（Whisper 检测为 {detected_lang}）")

    stem = src.stem
    video_info = {"title": stem, "channel": "", "upload_date": ""}

    print(f"\n✍️  开始 DeepSeek 重写（模型：{MODEL}）……")
    content = rewrite_to_chinese(raw_text, source_lang, is_asr=True, chunk_size=chunk_size)

    print("🏷️  生成摘要和标签……")
    try:
        meta = generate_metadata(stem, content)
        summary = meta.get("summary", "")
        tags = meta.get("tags", [])
    except Exception as e:
        print(f"⚠️  元数据生成失败（{e}）")
        summary = ""
        tags = []

    markdown = build_markdown(
        str(src), video_info, source_lang, summary, tags, content, is_local=True
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename:
        out_path = out_dir / output_filename
        if not out_path.suffix:
            out_path = out_path.with_suffix(".md")
    else:
        fname = safe_filename(stem)
        out_path = out_dir / f"{fname}.md"

    out_path.write_text(markdown, encoding="utf-8")

    print(f"\n{'═' * 60}")
    print(f"✅ 完成！输出文件：{out_path.resolve()}")
    print(f"{'═' * 60}\n")

    return str(out_path.resolve())


# ═══════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="将 YouTube 视频或本地媒体文件转为中文 Markdown 文档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（YouTube）：
  python youtube2markdown.py https://www.youtube.com/watch?v=xxxxx
  python youtube2markdown.py <URL> --force-whisper --whisper-model small
  python youtube2markdown.py <URL> --whisper-backend faster-whisper

示例（本地文件）：
  python youtube2markdown.py "/path/to/video.mp4"
  python youtube2markdown.py "./recording.mp4" -o meeting_notes.md
  python youtube2markdown.py "./audio.m4a" --whisper-model large
  python youtube2markdown.py "./video.mp4" --lang ja --whisper-backend mlx
        """,
    )
    parser.add_argument(
        "input",
        help="YouTube 视频 URL 或本地媒体文件路径（MP4/MKV/MOV/MP3/M4A 等）",
    )
    parser.add_argument("-o", "--output", dest="output_filename", default=None,
                        help="输出文件名（默认：根据视频标题/文件名自动生成）")
    parser.add_argument("--output-dir", default="output",
                        help="输出目录（默认：./output/）")
    parser.add_argument("--force-whisper", action="store_true",
                        help="强制使用 Whisper ASR（忽略已有字幕，仅 YouTube 模式有效）")
    parser.add_argument("--whisper-model",
                        choices=["tiny", "base", "small", "medium", "large"],
                        default="medium",
                        help="Whisper 模型大小（默认：medium）")
    parser.add_argument("--whisper-backend",
                        choices=["auto", "mlx", "faster-whisper", "torch"],
                        default="auto",
                        help=(
                            "转录后端（默认：auto）\n"
                            "  auto           自动选择：Apple Silicon→mlx / CUDA ct2→faster-whisper / CUDA torch→torch / 其他→CPU\n"
                            "  mlx            mlx-whisper，Apple Silicon GPU/ANE 加速（需安装）\n"
                            "  faster-whisper CPU/CUDA（自动检测，beam_size=1+VAD）\n"
                            "  torch          openai-whisper+torch，适用于 ARM64 CUDA 如 DGX Spark（需安装）"
                        ))
    parser.add_argument("--chunk-size", type=int, default=30000,
                        help="每次发送给 DeepSeek 的最大字符数（默认：30000）")
    parser.add_argument("--lang", dest="lang_hint", default=None,
                        help="手动指定源语言代码（如 en/ja/zh）")

    args = parser.parse_args()

    try:
        if is_local_file(args.input):
            process_local(
                file_path=args.input,
                output_dir=args.output_dir,
                output_filename=args.output_filename,
                whisper_model=args.whisper_model,
                whisper_backend=args.whisper_backend,
                chunk_size=args.chunk_size,
                lang_hint=args.lang_hint,
            )
        else:
            process(
                url=args.input,
                output_dir=args.output_dir,
                output_filename=args.output_filename,
                force_whisper=args.force_whisper,
                whisper_model=args.whisper_model,
                whisper_backend=args.whisper_backend,
                chunk_size=args.chunk_size,
            )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
