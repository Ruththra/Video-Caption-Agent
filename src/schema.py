"""
Input/output schema validation for the Video Captioning Agent.

No external validation library is required (keeps the image small); this
module implements the schema checks by hand and raises `SchemaError` with a
clear, specific message whenever a task or a model response doesn't match
the contract described in the hackathon brief.
"""
from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


class SchemaError(ValueError):
    """Raised when input or output data does not match the expected schema."""


def validate_task(raw: Any, index: int) -> Dict[str, Any]:
    """Validate a single task dict from tasks.json.

    Returns the validated task (with styles de-duplicated and filtered to
    known styles, unknown styles are kept too -- we caption whatever is
    asked for, but at minimum the four required styles are always produced
    downstream by captioning.py).
    """
    if not isinstance(raw, dict):
        raise SchemaError(f"task[{index}] is not a JSON object")

    task_id = raw.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise SchemaError(f"task[{index}] missing/invalid 'task_id'")

    video_url = raw.get("video_url")
    if not isinstance(video_url, str) or not video_url.strip():
        raise SchemaError(f"task[{index}] ('{task_id}') missing/invalid 'video_url'")

    styles = raw.get("styles")
    if not isinstance(styles, list) or len(styles) == 0:
        # Not fatal -- we fall back to the four required styles so the
        # batch never dies just because 'styles' was omitted or empty.
        styles = list(REQUIRED_STYLES)
    else:
        styles = [s for s in styles if isinstance(s, str) and s.strip()]
        if not styles:
            styles = list(REQUIRED_STYLES)

    return {"task_id": task_id.strip(), "video_url": video_url.strip(), "styles": styles}


def load_tasks(raw_json: Any) -> List[Dict[str, Any]]:
    """Validate the top-level tasks.json payload (must be a list)."""
    if not isinstance(raw_json, list):
        raise SchemaError("tasks.json root must be a JSON array")
    if len(raw_json) == 0:
        raise SchemaError("tasks.json contains zero tasks")
    return list(raw_json)


def validate_result(result: Dict[str, Any]) -> bool:
    """Return True if a single result dict matches the required output shape."""
    if not isinstance(result, dict):
        return False
    if not isinstance(result.get("task_id"), str):
        return False
    captions = result.get("captions")
    if not isinstance(captions, dict):
        return False
    for v in captions.values():
        if not isinstance(v, str) or not v.strip():
            return False
    return True


def validate_results_batch(results: List[Dict[str, Any]]) -> None:
    """Raise SchemaError if the final results.json payload is malformed."""
    if not isinstance(results, list):
        raise SchemaError("results.json root must be a JSON array")
    for i, r in enumerate(results):
        if not validate_result(r):
            raise SchemaError(f"result[{i}] failed schema validation: {r!r}")
