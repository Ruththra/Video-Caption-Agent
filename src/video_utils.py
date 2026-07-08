"""
Video download + ffmpeg/ffprobe frame sampling utilities.

Design goals:
  * Never throw an unhandled exception into main.py -- every public function
    either returns a usable result or raises one of the narrow exceptions
    below so main.py can catch-and-fallback per task.
  * Sample frames evenly across the *whole* clip duration, not just the
    first frame/second.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from typing import List

import requests

logger = logging.getLogger("video_utils")

DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "60"))
DOWNLOAD_RETRIES = int(os.environ.get("DOWNLOAD_RETRIES", "3"))
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(500 * 1024 * 1024)))  # 500MB safety cap
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "8"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "16"))
FFMPEG_TIMEOUT_SECONDS = int(os.environ.get("FFMPEG_TIMEOUT_SECONDS", "90"))


class VideoDownloadError(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    pass


def download_video(url: str, dest_dir: str) -> str:
    """Download `url` into `dest_dir` with retry/backoff. Returns local path."""
    last_err: Exception | None = None
    dest_path = os.path.join(dest_dir, "input_video.mp4")

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            logger.info("Downloading video (attempt %d/%d): %s", attempt, DOWNLOAD_RETRIES, url)
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
                resp.raise_for_status()
                written = 0
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > MAX_VIDEO_BYTES:
                            raise VideoDownloadError(
                                f"video exceeds MAX_VIDEO_BYTES ({MAX_VIDEO_BYTES})"
                            )
                        f.write(chunk)
                if written == 0:
                    raise VideoDownloadError("downloaded 0 bytes")
            return dest_path
        except VideoDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001 - we deliberately catch broadly here
            last_err = exc
            logger.warning("Download attempt %d failed: %s", attempt, exc)
            time.sleep(min(2 ** attempt, 8))

    raise VideoDownloadError(f"failed to download {url} after {DOWNLOAD_RETRIES} attempts: {last_err}")


def probe_duration_seconds(video_path: str) -> float:
    """Return video duration in seconds via ffprobe. Falls back to 60s on failure."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        duration = float(proc.stdout.strip())
        if duration <= 0:
            raise ValueError("non-positive duration")
        return duration
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffprobe failed (%s); assuming 60s fallback duration", exc)
        return 60.0


def _frame_count_for_duration(duration: float) -> int:
    # Roughly one frame every ~6-8 seconds, clamped to [MIN_FRAMES, MAX_FRAMES].
    est = round(duration / 7.0)
    return max(MIN_FRAMES, min(MAX_FRAMES, est if est > 0 else MIN_FRAMES))


def extract_frames(video_path: str, out_dir: str, max_frames_override: int | None = None) -> List[str]:
    """Extract evenly-spaced JPEG frames spanning the whole video.

    Returns a chronologically sorted list of file paths. Raises FFmpegError
    if no frames could be produced at all.
    """
    duration = probe_duration_seconds(video_path)
    n_frames = max_frames_override or _frame_count_for_duration(duration)

    # Evenly-spaced timestamps, inset slightly from the very start/end so we
    # don't land on black frames/fades.
    inset = min(0.5, duration * 0.03)
    span = max(duration - 2 * inset, 0.1)
    timestamps = [inset + span * (i / max(n_frames - 1, 1)) for i in range(n_frames)]

    frame_paths: List[str] = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
                    "-frames:v", "1", "-q:v", "3", out_path,
                ],
                capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS, check=True,
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                frame_paths.append(out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ffmpeg failed extracting frame at %.2fs: %s", ts, exc)
            continue

    if not frame_paths:
        raise FFmpegError("no frames could be extracted from video")

    return sorted(frame_paths)


def cleanup_dir(path: str) -> None:
    try:
        for root, _dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
    except Exception as exc:  # noqa: BLE001
        logger.debug("cleanup failed for %s: %s", path, exc)


def make_scratch_dir() -> str:
    return tempfile.mkdtemp(prefix="vcagent_")
