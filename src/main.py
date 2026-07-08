#!/usr/bin/env python3
"""
Video Captioning Agent -- entrypoint.

Reads   /input/tasks.json
Writes  /output/results.json
Always exits 0 as long as it managed to write *some* valid results.json,
per the hackathon contract. Per-task failures are caught and turned into
best-effort fallback captions rather than crashing the batch.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from captioning import REQUIRED_STYLES, generate_captions_for_task, template_fallback_caption
from model_client import ModelError, build_client, encode_jpeg_files
from schema import SchemaError, load_tasks, validate_results_batch, validate_task
from video_utils import (
    FFmpegError,
    VideoDownloadError,
    cleanup_dir,
    download_video,
    extract_frames,
    make_scratch_dir,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # never write logs to stdout
)
logger = logging.getLogger("main")

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "16"))


def write_results(results: list) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)  # atomic-ish swap so a crash mid-write can't corrupt output
    logger.info("Wrote %d result(s) to %s", len(results), OUTPUT_PATH)


def fallback_result(task_id: str, styles: list, reason: str) -> dict:
    """Used when a task fails so badly we never got a scene summary at all."""
    logger.error("task '%s' falling back to generic captions: %s", task_id, reason)
    generic_summary = {
        "setting": "unspecified (processing failed)",
        "subjects": [],
        "actions": [],
        "objects": [],
        "mood": "neutral",
        "notable_details": "",
    }
    captions = {s: template_fallback_caption(generic_summary, s) for s in styles}
    for s in REQUIRED_STYLES:
        captions.setdefault(s, template_fallback_caption(generic_summary, s))
    return {"task_id": task_id, "captions": captions}


def process_task(client, raw_task: dict, index: int) -> dict:
    # 1. Validate shape first -- if task_id itself is missing we can't even
    #    build a meaningful fallback result, so we synthesize a placeholder id.
    try:
        task = validate_task(raw_task, index)
    except SchemaError as exc:
        placeholder_id = raw_task.get("task_id") if isinstance(raw_task, dict) else None
        placeholder_id = placeholder_id if isinstance(placeholder_id, str) and placeholder_id else f"unknown_{index}"
        return fallback_result(placeholder_id, REQUIRED_STYLES, f"schema error: {exc}")

    task_id, video_url, styles = task["task_id"], task["video_url"], task["styles"]
    scratch = make_scratch_dir()
    try:
        try:
            video_path = download_video(video_url, scratch)
        except VideoDownloadError as exc:
            return fallback_result(task_id, styles, f"download failed: {exc}")

        try:
            frame_paths = extract_frames(video_path, scratch, max_frames_override=MAX_FRAMES)
        except FFmpegError as exc:
            return fallback_result(task_id, styles, f"frame extraction failed: {exc}")

        images_b64 = encode_jpeg_files(frame_paths)

        try:
            captions = generate_captions_for_task(client, images_b64, styles)
        except ModelError as exc:
            return fallback_result(task_id, styles, f"model inference failed: {exc}")

        return {"task_id": task_id, "captions": captions}
    except Exception as exc:  # noqa: BLE001 - final safety net, batch must never die
        return fallback_result(task_id, styles, f"unexpected error: {exc}")
    finally:
        cleanup_dir(scratch)
        try:
            os.rmdir(scratch)
        except OSError:
            pass


def main() -> int:
    start = time.time()
    results: list = []

    if not os.path.exists(INPUT_PATH):
        logger.error("Input file not found at %s", INPUT_PATH)
        write_results([])
        return 0

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        raw_tasks = load_tasks(raw_json)
    except (json.JSONDecodeError, SchemaError) as exc:
        logger.error("tasks.json is malformed: %s", exc)
        write_results([])
        return 0

    try:
        client = build_client()
    except ModelError as exc:
        logger.error("Could not initialize model client: %s", exc)
        # Still produce best-effort template captions for every task rather
        # than exiting empty-handed.
        for i, raw_task in enumerate(raw_tasks):
            tid = raw_task.get("task_id") if isinstance(raw_task, dict) else None
            tid = tid if isinstance(tid, str) and tid else f"unknown_{i}"
            styles = raw_task.get("styles") if isinstance(raw_task, dict) else None
            styles = styles if isinstance(styles, list) and styles else REQUIRED_STYLES
            results.append(fallback_result(tid, styles, f"no model client: {exc}"))
        write_results(results)
        return 0

    for i, raw_task in enumerate(raw_tasks):
        logger.info("Processing task %d/%d", i + 1, len(raw_tasks))
        try:
            results.append(process_task(client, raw_task, i))
        except Exception as exc:  # noqa: BLE001 - absolute last resort per task
            logger.error("task[%d] crashed outside process_task: %s", i, exc)
            tid = raw_task.get("task_id") if isinstance(raw_task, dict) else f"unknown_{i}"
            results.append(fallback_result(tid or f"unknown_{i}", REQUIRED_STYLES, str(exc)))

    try:
        validate_results_batch(results)
    except SchemaError as exc:
        logger.error("Final validation failed (%s); writing anyway for partial credit", exc)

    write_results(results)
    logger.info("Done in %.1fs", time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
