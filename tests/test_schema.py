import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from schema import (  # noqa: E402
    REQUIRED_STYLES,
    SchemaError,
    load_tasks,
    validate_result,
    validate_results_batch,
    validate_task,
)


def test_validate_task_ok():
    raw = {"task_id": "v1", "video_url": "https://x.com/v.mp4", "styles": ["formal"]}
    task = validate_task(raw, 0)
    assert task["task_id"] == "v1"
    assert task["styles"] == ["formal"]


def test_validate_task_missing_id_raises():
    with pytest.raises(SchemaError):
        validate_task({"video_url": "https://x.com/v.mp4"}, 0)


def test_validate_task_missing_url_raises():
    with pytest.raises(SchemaError):
        validate_task({"task_id": "v1"}, 0)


def test_validate_task_empty_styles_falls_back_to_required():
    raw = {"task_id": "v1", "video_url": "https://x.com/v.mp4", "styles": []}
    task = validate_task(raw, 0)
    assert task["styles"] == REQUIRED_STYLES


def test_validate_task_not_a_dict_raises():
    with pytest.raises(SchemaError):
        validate_task("not a dict", 0)


def test_load_tasks_requires_list():
    with pytest.raises(SchemaError):
        load_tasks({"not": "a list"})


def test_load_tasks_requires_nonempty():
    with pytest.raises(SchemaError):
        load_tasks([])


def test_validate_result_ok():
    result = {"task_id": "v1", "captions": {"formal": "A clip of a street."}}
    assert validate_result(result) is True


def test_validate_result_rejects_empty_caption():
    result = {"task_id": "v1", "captions": {"formal": ""}}
    assert validate_result(result) is False


def test_validate_result_rejects_missing_captions():
    assert validate_result({"task_id": "v1"}) is False


def test_validate_results_batch_ok():
    validate_results_batch([{"task_id": "v1", "captions": {"formal": "ok"}}])


def test_validate_results_batch_raises_on_bad_entry():
    with pytest.raises(SchemaError):
        validate_results_batch([{"task_id": "v1"}])
