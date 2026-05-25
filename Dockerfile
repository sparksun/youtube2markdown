# ── 基础镜像：NVIDIA 官方 PyTorch（包含 ARM64+CUDA 完整支持）────
# 版本列表：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
ARG PYTORCH_VERSION=25.11-py3
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}

LABEL maintainer="youtube2markdown"
LABEL description="YouTube/本地视频 → 中文 Markdown，GPU 加速转录"

WORKDIR /app

# ── 系统依赖：用 conda-forge 安装 ffmpeg（比 apt 更可靠，有 ARM64 预编译包）──
# NVIDIA PyTorch 容器自带 conda，conda-forge 包含 ARM64 静态编译的 ffmpeg
RUN conda install -y -c conda-forge ffmpeg && conda clean -ya

# ── Python 依赖 ───────────────────────────────────────────────────
# 基础镜像已含 torch，只装其余依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 转录后端：openai-whisper（torch backend，适配 ARM64 CUDA）
RUN pip install --no-cache-dir openai-whisper

# ── 复制代码 ──────────────────────────────────────────────────────
COPY youtube2markdown.py .
COPY .env.example .

# ── 持久化目录 ────────────────────────────────────────────────────
RUN mkdir -p input output
VOLUME ["/app/input", "/app/output"]

# ── 入口 ──────────────────────────────────────────────────────────
ENTRYPOINT ["python", "youtube2markdown.py"]
CMD ["--help"]
