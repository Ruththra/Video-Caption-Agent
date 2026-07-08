#!/usr/bin/env python3
"""Direct one-frame Fireworks/Kimi debug call.

Usage:
  FIREWORKS_API_KEY=... \
  MODEL_PROVIDER=fireworks \
  MODEL_NAME=accounts/fireworks/models/kimi-k2p6 \
  python scripts/debug_kimi_one_frame.py https://example.com/video.mp4
"""
from __future__ import annotations

import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from model_client import FireworksClient, encode_jpeg_files  # noqa: E402
from video_utils import download_video, extract_frames, make_scratch_dir, cleanup_dir  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PROMPT = """Describe this image frame factually in one compact paragraph.
Only mention visible subjects, setting, actions, objects, and mood.
Do not return JSON. Do not use markdown."""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_kimi_one_frame.py <video_url>", file=sys.stderr)
        return 2

    scratch = make_scratch_dir()
    try:
        video_path = download_video(sys.argv[1], scratch)
        frame_paths = extract_frames(video_path, scratch, max_frames_override=1)
        images_b64 = encode_jpeg_files(frame_paths)
        client = FireworksClient()
        raw = client.caption(images_b64, PROMPT, json_mode=False)
        print("\n=== FRAME PATH ===")
        print(frame_paths[0])
        print("\n=== RAW MODEL TEXT ===")
        print(raw)
        print("\n=== RAW RESPONSE OBJECT WRITTEN? ===")
        print("Set MODEL_RAW_LOG_DIR=/tmp/model_raw to preserve full Fireworks response JSON files.")
        return 0
    finally:
        cleanup_dir(scratch)
        try:
            os.rmdir(scratch)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
