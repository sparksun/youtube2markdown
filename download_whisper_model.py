#!/usr/bin/env python3
"""
在 Docker build 阶段预下载 Whisper 模型。
使用 wget -c（断点续传）+ 最多 10 次重试，避免网络抖动导致的下载失败。
SHA256 从 openai-whisper 内置的模型 URL 路径中提取并校验。
"""
import sys
import os
import hashlib
import subprocess
import time

import whisper


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(model_name: str, max_retries: int = 10, retry_delay: int = 10):
    if model_name not in whisper._MODELS:
        print(f"❌ 未知模型：{model_name}，可选：{list(whisper._MODELS.keys())}")
        sys.exit(1)

    url = whisper._MODELS[model_name]
    expected_sha256 = url.split("/")[-2]   # URL 路径倒数第二段即 SHA256
    filename = url.split("/")[-1]

    cache_dir = os.path.expanduser("~/.cache/whisper")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, filename)

    print(f"📥 模型：{model_name}  →  {dst}")
    print(f"🔗 URL：{url}")

    for attempt in range(1, max_retries + 1):
        print(f"\n⏳ 第 {attempt}/{max_retries} 次尝试……")
        result = subprocess.run(
            [
                "wget",
                "-c",                        # 断点续传
                "--retry-connrefused",
                "--tries=3",
                "--timeout=60",
                "--waitretry=5",
                "-q", "--show-progress",
                url, "-O", dst,
            ]
        )

        if result.returncode != 0:
            print(f"⚠️  wget 失败（code {result.returncode}），{retry_delay}s 后重试……")
            time.sleep(retry_delay)
            continue

        actual_sha256 = sha256_file(dst)
        if actual_sha256 == expected_sha256:
            size_mb = os.path.getsize(dst) / 1024 / 1024
            print(f"✅ Whisper [{model_name}] 模型已缓存（{size_mb:.0f} MB）")
            return

        print(f"⚠️  SHA256 校验失败（期望 {expected_sha256[:16]}…，实际 {actual_sha256[:16]}…）")
        os.remove(dst)
        print(f"   已删除损坏文件，{retry_delay}s 后重试……")
        time.sleep(retry_delay)

    print(f"❌ 下载失败，已重试 {max_retries} 次，放弃。")
    sys.exit(1)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "medium"
    main(model)
