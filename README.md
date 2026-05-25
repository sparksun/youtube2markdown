# youtube2markdown

将 YouTube 视频或**本地媒体文件**（MP4、MKV、MP3 等）自动转换为通顺的中文 Markdown 文档。

## 功能

- 🎬 支持 YouTube 视频（英文、日文、中文）
- 📂 支持本地媒体文件（MP4 / MKV / MOV / AVI / MP3 / M4A 等）
- 📝 YouTube 模式：优先使用自带字幕（秒级获取）
- 🎙️ 无字幕 / 本地文件：Whisper ASR 转录（含实时进度条）
- ⚡ 智能转录后端自动选择：Apple Silicon → **mlx-whisper**（5–10×）；NVIDIA GPU → **CUDA float16**（~10–20×）；CPU int8 备用
- 🤖 通过 OpenRouter → DeepSeek 纠错、翻译、重写为通顺中文
- 📄 输出带 YAML frontmatter 的 Markdown，可直接导入 Obsidian

## 快速开始

### 1. 安装依赖

```bash
cd youtube2markdown
pip install -r requirements.txt

# 本地视频/音频文件 或 YouTube 无字幕回退时需要：
pip install faster-whisper
brew install ffmpeg  # macOS；Ubuntu: sudo apt install ffmpeg
```

**Apple Silicon Mac（M系列芯片）——推荐额外安装 mlx-whisper（5–10× 加速）**

```bash
pip install mlx-whisper
# 首次运行会自动下载模型（medium ≈ 1.5 GB）
# 无需额外配置，auto 模式下会自动优先使用
```

**NVIDIA GPU（Windows / Linux）——自动启用 CUDA 加速**

```bash
# ctranslate2 4.x 已内置 CUDA 支持，无需 [cuda] extra
pip install ctranslate2 faster-whisper
# 安装后 faster-whisper 会自动识别 NVIDIA GPU 并使用 float16 加速
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 OpenRouter API Key：

```bash
cp .env.example .env
# 编辑 .env，填入 OPENROUTER_API_KEY
```

### 3. 运行

**NVIDIA DGX Spark / Grace Blackwell（推荐使用 Docker）**

ctranslate2 目前无 ARM64+CUDA 官方 wheel，最省心的方案是直接使用 NVIDIA 官方 PyTorch 镜像，
其中 CUDA 栈已为 ARM64 完整预编译：

```bash
# 首次构建镜像（约 5–10 分钟，仅需一次）
docker compose build

# 处理本地文件
docker compose run --rm youtube2markdown "/app/input/video.mp4"

# 处理 YouTube URL
docker compose run --rm youtube2markdown https://www.youtube.com/watch?v=xxxxx

# 指定参数
docker compose run --rm youtube2markdown "/app/input/video.mp4" \
  --whisper-model medium --output-dir /app/output
```

> 将媒体文件放入本机 `input/` 目录，输出结果在本机 `output/` 目录。
> `.env` 文件中的 `OPENROUTER_API_KEY` 会自动注入容器。
> 容器内 `auto` 模式会自动选择 `torch` GPU 后端。

---

#### YouTube 视频

```bash
# 基本用法（自动获取字幕，无字幕时回退 Whisper）
python youtube2markdown.py https://www.youtube.com/watch?v=xxxxx

# 指定输出文件名
python youtube2markdown.py <URL> -o my_notes.md

# 指定输出目录
python youtube2markdown.py <URL> --output-dir ~/Documents/notes/

# 强制使用 Whisper（忽略已有字幕）、指定后端
python youtube2markdown.py <URL> --force-whisper --whisper-model small
python youtube2markdown.py <URL> --force-whisper --whisper-backend mlx
```

#### 本地媒体文件

```bash
# 处理本地 MP4 视频（自动提取音频 → Whisper 转录 → DeepSeek 重写）
python youtube2markdown.py "/path/to/video.mp4"

# 指定输出文件名
python youtube2markdown.py "./recording.mp4" -o meeting_notes.md

# 处理音频文件（MP3/M4A，跳过 ffmpeg 步骤，直接转录）
python youtube2markdown.py "./audio.m4a" --whisper-model large

# 手动指定语言（覆盖 Whisper 自动检测）
python youtube2markdown.py "./meeting.mp4" --lang ja

# 强制指定转录后端
python youtube2markdown.py "./video.mp4" --whisper-backend mlx          # Apple Silicon GPU
python youtube2markdown.py "./video.mp4" --whisper-backend faster-whisper # 强制 CPU/CUDA
```

> **支持的本地文件格式**  
> 视频：`.mp4` `.mkv` `.mov` `.avi` `.webm`  
> 音频：`.mp3` `.m4a` `.aac` `.wav` `.flac` `.ogg`

## 处理流程

```
YouTube URL ──→ 字幕 API ─────────────────────────────────────┐
               （无字幕）→ yt-dlp 下载音频 ──────────────────┤
                                                              ├─→ Whisper 转录 → DeepSeek 重写 → Markdown
本地 MP4 ──→ ffmpeg 提取音频 ──────────────────────────┤
本地 MP3 ──→ 直接转录 ─────────────────────────────────┘

Whisper 后端自动选择（--whisper-backend auto）：
  Apple Silicon + mlx-whisper 已安装  →  mlx   （GPU/ANE，5–10×）
  NVIDIA CUDA GPU 可用            →  faster-whisper （float16，~10–20×）
  其他 CPU                         →  faster-whisper （int8）
```

## 输出格式

**YouTube 视频**：

```markdown
---
title: "视频标题"
source: "https://youtube.com/watch?v=..."
channel: "频道名"
date: "2026-05-25"
language: "en"
tags:
  - "AI"
  - "机器学习"
generated: "2026-05-25"
tool: "youtube2markdown"
---

# 视频标题

> **摘要**：xxx

---

整理后的中文正文……

---
*本文由 youtube2markdown 自动生成 · 原视频语言：英文 · 生成日期：2026-05-25*
```

**本地文件**（`source` 改为 `file` 字段）：

```markdown
---
title: "recording"
file: "recording.mp4"
date: "2026-05-25"
language: "ja"
tags:
  - "会议"
generated: "2026-05-25"
tool: "youtube2markdown"
---

# recording

> **摘要**：xxx

---

整理后的中文正文……

---
*本文由 youtube2markdown 自动生成 · 来源：本地文件 · 原始语言：日文 · 生成日期：2026-05-25*
```

## 选项说明

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output` | 输出文件名 | 根据视频标题/文件名自动生成 |
| `--output-dir` | 输出目录 | `./output/` |
| `--force-whisper` | 强制 Whisper ASR（仅 YouTube 模式，忽略已有字幕） | 否 |
| `--whisper-model` | Whisper 模型大小（`tiny` / `base` / `small` / `medium` / `large`） | `medium` |
| `--whisper-backend` | 转录后端：`auto` / `mlx` / `faster-whisper`（见下文） | `auto` |
| `--chunk-size` | 每次发送给 DeepSeek 的最大字符数 | `30000` |
| `--lang` | 手动指定源语言代码（如 `en` / `ja` / `zh`），本地文件模式下覆盖 Whisper 自动检测 | 自动检测 |

## 转录后端选择（--whisper-backend）

| 后端 | 设备 | 模式 | 预估加速 | 适用场景 |
|------|------|------|---------|----------|
| `mlx` | Apple Silicon GPU/ANE | float16 | **5–10×** | Mac M系列，需 `pip install mlx-whisper` |
| `faster-whisper` (CUDA) | NVIDIA GPU | float16 | **10–20×** | x86_64 Linux/Windows 带 CUDA，自动检测 |
| `torch` | NVIDIA GPU (ARM64) | float16 | **类似 GPU** | DGX Spark / Jetson 等 ARM64 CUDA，需 `pip install openai-whisper torch` |
| `faster-whisper` (CPU) | CPU | int8 | 基准 | 无 GPU 时自动回退 |

`auto`（默认）自动检测和选择，优先级：**mlx → faster-whisper（CUDA）→ torch（CUDA）→ CPU**。

## Whisper 转录进度

**mlx-whisper（Apple Silicon）**：无段迭代，spinner 显示进度：

```
🚀 [mlx-whisper / medium] Apple Silicon 加速转录……
   模型仓库：mlx-community/whisper-medium-mlx（首次运行会自动下载）
  ⠻  转录中……  已用 38s
✅ 转录完成，耗时 42.1s
```

**openai-whisper + torch（DGX Spark / ARM64 CUDA）**：GPU 加速，spinner 显示进度：

```
  🔍 自动选择转录后端：torch
  ℹ️  ctranslate2 无 CUDA 支持（ARM64 平台），改用 openai-whisper + torch GPU
🚀 [openai-whisper / medium / NVIDIA GPU] 加载模型……
🎙️  开始转录（首次运行不需下载）……
  ⠼  转录中……  已用 62s
✅ 转录完成，耗时 68.3s
```

**faster-whisper（CUDA GPU / CPU）**：逐段迭代，显示详细进度条：

```
🎙️  [faster-whisper / medium / NVIDIA GPU / float16] 转录中（beam_size=1, VAD 已启用）……
  [████████░░░░░░░░░░░░]  40.2%  12:05 / 30:00  已用 18s  预计剩余 27s
  [████████████░░░░░░░░]  60.3%  18:05 / 30:00  已用 26s  预计剩余 17s
✅ 转录完成，共 312 段，耗时 44.3s
```

## 环境要求

- Python 3.10+
- OpenRouter API Key（[获取地址](https://openrouter.ai/settings/keys)）
- `ffmpeg`（处理本地视频文件时必须；macOS: `brew install ffmpeg`）
- `faster-whisper`（本地文件 / YouTube 无字幕回退）
- `mlx-whisper`（可选，Apple Silicon 专属；`pip install mlx-whisper`）
- `openai-whisper` + `torch`（可选，ARM64 CUDA 如 DGX Spark；`pip install openai-whisper torch`）
- CUDA（可选，x86_64 NVIDIA GPU；`pip install ctranslate2 faster-whisper`，4.x 版已内置 CUDA）

> **DGX Spark / Grace Blackwell 用户**：
> ctranslate2 目前无 ARM64+CUDA 官方 wheel，请改用 `torch` 后端：
> ```bash
> pip install openai-whisper torch
> # auto 模式下会自动选择 torch GPU
> ```

## 未来计划

- [ ] Apple Podcasts 支持（RSS Feed → 音频 → 中文 Markdown）
- [ ] 批量处理播放列表 / 目录下所有文件
- [ ] 自定义提示词模板
