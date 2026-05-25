# ── 基础镜像：NVIDIA 官方 PyTorch（包含 ARM64+CUDA 完整支持）────
# 版本列表：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
ARG PYTORCH_VERSION=25.11-py3
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}

LABEL maintainer="youtube2markdown"
LABEL description="YouTube/本地视频 → 中文 Markdown，GPU 加速转录"

WORKDIR /app

# ── ffmpeg：用 imageio-ffmpeg（pip 包，内含 ARM64 静态二进制，无需 apt/conda）──
RUN pip install --no-cache-dir imageio-ffmpeg && \
    ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg

# ── Python 依赖 ───────────────────────────────────────────────────
# 基础镜像已含 torch，只装其余依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 转录后端：openai-whisper（torch backend，适配 ARM64 CUDA）
RUN pip install --no-cache-dir openai-whisper

# ── 预下载 Whisper 模型（烤进镜像，运行时完全离线）────────────────
# 使用带断点续传和重试的脚本，避免网络抖动导致 SHA256 校验失败
# medium ≈ 1.4 GB；small ≈ 461 MB；构建时通过 --build-arg WHISPER_MODEL=small 指定
ARG WHISPER_MODEL=medium
COPY download_whisper_model.py /tmp/download_whisper_model.py
RUN python /tmp/download_whisper_model.py "${WHISPER_MODEL}"

# ── 复制代码 ──────────────────────────────────────────────────────
COPY youtube2markdown.py .
COPY .env.example .

# ── 持久化目录 ────────────────────────────────────────────────────
RUN mkdir -p input output
VOLUME ["/app/input", "/app/output"]

# ── 入口 ──────────────────────────────────────────────────────────
ENTRYPOINT ["python", "youtube2markdown.py"]
CMD ["--help"]
