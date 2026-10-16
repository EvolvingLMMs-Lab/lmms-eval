"""Keep the upstream cache contract for unrelated benchmark requests."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from lmms_eval.api.instance import GenerationResult, Instance
from lmms_eval.caching.fs_detect import FsType
from lmms_eval.caching.response_cache import ResponseCache


@pytest.mark.parametrize(
    "text",
    [
        "B",
        "[LMMS_EVAL_REQUEST_FAILED after 1 retries] timeout",
        "[LMMS_EVAL_BUDGET_EXCEEDED]",
        json.dumps({"api_response": {"status_code": 503, "body": {"error": "failure"}}}),
        json.dumps({"api_response": {"status_code": 400, "body": {"error": {"code": "data_inspection_failed"}}}}),
    ],
)
def test_generic_strings_keep_upstream_validity(text):
    assert ResponseCache._is_valid_response(text, "generate_until", "mme")
    assert ResponseCache._is_valid_response(GenerationResult(text=text), "generate_until", "mme")


def test_non_omnicompass_scratch_layout_and_resume_are_preserved(tmp_path):
    root = tmp_path / "shared"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    task = SimpleNamespace(dump_config=lambda: {"task": "mme"})
    with (
        patch("lmms_eval.caching.response_cache.detect_fs_type", return_value=FsType.REMOTE),
        patch("lmms_eval.caching.response_cache.find_local_scratch", return_value=str(scratch)),
        patch("lmms_eval.caching.response_cache._resolve_cache_run_id", return_value="existing-job"),
        patch("lmms_eval.tasks.omnicompass._core.run_identity.model_identity", side_effect=AssertionError("OmniCompass policy used")),
    ):
        cache = ResponseCache.create(str(root), model="fixture", task_dict={"mme": task})
    try:
        parts = Path(cache.db_path).relative_to(scratch).parts
        assert parts[0] == "lmms_eval_cache"
        assert parts[2:] == ("runs", "existing-job", "rank_0.db")
        assert not cache._strict_checkpoint
        assert cache._writer_lock is None
        req = Instance(request_type="generate_until", arguments=("Question", {"temperature": 0}, None, 0, "mme", "test"), idx=0, metadata={"task": "mme", "doc_id": 0, "repeats": 1})
        model = SimpleNamespace(generate_until=Mock(return_value=["B"]))
        assert cache.execute(model, "generate_until", [req]) == ["B"]
        assert cache.execute(model, "generate_until", [req]) == ["B"]
        model.generate_until.assert_called_once()
        with patch("lmms_eval.caching.response_cache.shutil.copy2", wraps=shutil.copy2) as copy:
            cache._checkpoint_to_run_dir()
            assert copy.call_count == 2
    finally:
        cache.close()
