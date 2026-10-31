import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.caching import response_cache as module
from lmms_eval.caching.fs_detect import FsType
from lmms_eval.caching.response_cache import ResponseCache


def request():
    return Instance(request_type="generate_until", arguments=("prompt", {"temperature": 0}, None, 0, "t", "test"), idx=0, metadata={"task": "t", "doc_id": 0, "repeats": 1})


class Legacy:
    def __init__(self, answer):
        self.generate_until = Mock(return_value=[answer])


class Identified(Legacy):
    def __init__(self, answer, identity):
        super().__init__(answer)
        self.identity = identity

    def response_cache_identity(self):
        return self.identity


def run(root, model, model_args="pretrained=weights"):
    cache = ResponseCache.create(str(root), model="adapter", model_args=model_args, model_object=model)
    result = cache.execute(model, "generate_until", [request()])
    stats = cache.get_stats()
    cache.finalize(success=True)
    return result, stats


def test_real_cache_identity_hit_miss_and_legacy_isolation(tmp_path):
    root = tmp_path / "cache"
    assert run(root, Legacy("legacy"))[0] == ["legacy"]
    assert run(root, Legacy("unused"))[0] == ["legacy"]
    base = {"commit": "a", "dtype": "float16", "attention": "sdpa"}
    print("LMMS1599_IDENTITY_SEAM_READY")
    assert run(root, Identified("first", base))[0] == ["first"]
    assert run(root, Identified("unused", dict(reversed(list(base.items())))), model_args={"pretrained": "weights"})[0] == ["first"]
    for field, changed in (("commit", "b"), ("dtype", "bfloat16"), ("attention", "eager")):
        identity = {**base, field: changed}
        assert run(root, Identified(field, identity))[0] == [field]
    assert run(root, Identified("unused", base))[0] == ["first"]
    assert run(root, Legacy("unused"))[0] == ["legacy"]
    db = sqlite3.connect(root / "cache.db")
    assert db.execute("SELECT count(*) FROM responses").fetchone()[0] == 5
    assert json.loads(db.execute("SELECT value FROM meta WHERE key='model_identity'").fetchone()[0]) is None
    assert "effective_identity_v1" not in db.execute("SELECT value FROM meta WHERE key='model_fingerprint'").fetchone()[0]
    db.close()


@pytest.mark.parametrize("identity", [{}, [], {"bad": float("nan")}, {"bad": object()}])
def test_invalid_provider_creates_no_cache_artifacts(tmp_path, identity):
    root = tmp_path / "cache"
    with pytest.raises((ValueError, TypeError)):
        ResponseCache.create(str(root), model_object=Identified("x", identity))
    assert not root.exists()


def test_provider_exception_creates_no_cache_artifacts(tmp_path):
    class Broken(Legacy):
        def response_cache_identity(self):
            raise RuntimeError("provider failed")

    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="provider failed"):
        ResponseCache.create(str(root), model_object=Broken("x"))
    assert not root.exists()


def test_base_default_keeps_legacy_fingerprint(tmp_path):
    class Default(lmms):
        generate_until = Legacy("x").generate_until

        def loglikelihood(self, requests):
            return []

        def generate_until_multi_round(self, requests):
            return []

    model = Default()
    assert model.response_cache_identity() is None
    assert run(tmp_path / "cache", Legacy("legacy"))[0] == ["legacy"]
    assert run(tmp_path / "cache", model)[0] == ["legacy"]


def test_explicit_none_override_uses_legacy_key(tmp_path):
    root = tmp_path / "cache"
    assert run(root, Legacy("legacy"))[0] == ["legacy"]
    assert run(root, Identified("unused", None))[0] == ["legacy"]


def test_old_constructor_positional_arguments_and_detached_identity(tmp_path):
    db_path = tmp_path / "cache.db"
    audit_path = tmp_path / "cache.audit.jsonl"
    old = ResponseCache(str(db_path), str(audit_path), "adapter", {"t": "fingerprint"}, None, "version")
    assert old._task_fingerprints == {"t": "fingerprint"}
    assert old._eval_version == "version"
    old.close()
    identity = {"nested": {"revision": "before"}}
    cache = ResponseCache(str(db_path), str(audit_path), model_identity=identity)
    identity["nested"]["revision"] = "after"
    model = Legacy("answer")
    cache.execute(model, "generate_until", [request()])
    cache.close()
    with open(audit_path) as handle:
        assert json.loads(next(handle))["model_identity"] == {"nested": {"revision": "before"}}


def test_bad_direct_constructor_identity_creates_no_database(tmp_path):
    db_path = tmp_path / "cache.db"
    with pytest.raises((ValueError, TypeError)):
        ResponseCache(str(db_path), str(tmp_path / "audit.jsonl"), model_identity={"bad": float("nan")})
    assert not db_path.exists()


def test_identity_snapshot_in_meta_and_audit(tmp_path):
    identity = {"nested": {"z": 1, "a": 2}}
    root = tmp_path / "cache"
    assert run(root, Identified("answer", identity))[0] == ["answer"]
    identity["nested"]["z"] = 99
    db = sqlite3.connect(root / "cache.db")
    assert json.loads(db.execute("SELECT value FROM meta WHERE key='model_identity'").fetchone()[0]) == {"nested": {"a": 2, "z": 1}}
    db.close()
    with open(root / "cache.audit.jsonl") as handle:
        assert json.loads(next(handle))["model_identity"] == {"nested": {"a": 2, "z": 1}}


def test_identity_isolation_across_nfs_scratch_resume_and_publication(tmp_path, monkeypatch):
    remote, local = tmp_path / "remote", tmp_path / "local"
    local.mkdir()
    identity_a, identity_b = {"commit": "a"}, {"commit": "b"}
    monkeypatch.setenv("LMMS_CACHE_RUN_ID", "first")
    with (
        patch.object(module, "detect_fs_type", side_effect=lambda path: FsType.REMOTE if str(path).startswith(str(remote)) else FsType.LOCAL),
        patch.object(module, "find_local_scratch", return_value=str(local)) as scratch,
    ):
        first_model = Identified("a", identity_a)
        first = ResponseCache.create(str(remote), model="adapter", model_args="pretrained=weights", model_object=first_model)
        assert first.execute(first_model, "generate_until", [request()]) == ["a"]
        assert first.get_stats()["misses"] == 1
        first._checkpoint_to_run_dir()
        assert (remote / "runs/first/rank_0.db").exists()
        first_db_path = first.db_path
        first.close()

        other_local = tmp_path / "other-local"
        other_local.mkdir()
        scratch.return_value = str(other_local)
        resumed_model = Identified("unused", identity_a)
        resumed = ResponseCache.create(str(remote), model="adapter", model_args="pretrained=weights", model_object=resumed_model)
        assert resumed.db_path != first_db_path
        assert resumed.execute(resumed_model, "generate_until", [request()]) == ["a"]
        assert resumed.get_stats()["hits"] == 1
        assert resumed.get_stats()["hits_shared"] == 0
        resumed_model.generate_until.assert_not_called()
        resumed.finalize(success=True)

        with sqlite3.connect(remote / "cache.db") as db:
            assert json.loads(db.execute("SELECT value FROM meta WHERE key='model_identity'").fetchone()[0]) == identity_a

        monkeypatch.setenv("LMMS_CACHE_RUN_ID", "second")
        shared_model = Identified("unused", identity_a)
        shared = ResponseCache.create(str(remote), model="adapter", model_args="pretrained=weights", model_object=shared_model)
        assert shared.execute(shared_model, "generate_until", [request()]) == ["a"]
        assert shared.get_stats()["hits_shared"] == 1
        shared_model.generate_until.assert_not_called()
        shared.finalize(success=True)

        monkeypatch.setenv("LMMS_CACHE_RUN_ID", "third")
        b_result, b_stats = run(remote, Identified("b", identity_b))
        assert b_result == ["b"]
        assert b_stats["misses"] == 1
        monkeypatch.setenv("LMMS_CACHE_RUN_ID", "fourth")
        legacy_result, legacy_stats = run(remote, Legacy("legacy"))
        assert legacy_result == ["legacy"]
        assert legacy_stats["misses"] == 1

    with sqlite3.connect(remote / "cache.db") as db:
        assert db.execute("SELECT count(*) FROM responses").fetchone()[0] == 3
        assert json.loads(db.execute("SELECT value FROM meta WHERE key='model_identity'").fetchone()[0]) is None
    with open(remote / "cache.audit.jsonl") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert {json.loads(record["response"]): record.get("model_identity") for record in records} == {
        "a": identity_a,
        "b": identity_b,
        "legacy": None,
    }


def test_provider_called_once_before_execution_and_snapshot_stays_fixed(tmp_path):
    identity = {"revision": "before"}
    model = Identified("answer", identity)
    model.response_cache_identity = Mock(wraps=model.response_cache_identity)
    cache = ResponseCache.create(str(tmp_path / "cache"), model_object=model)
    identity["revision"] = "after"
    cache.execute(model, "generate_until", [request()])
    cache.finalize(success=True)
    model.response_cache_identity.assert_called_once_with()
    with open(tmp_path / "cache" / "cache.audit.jsonl") as handle:
        assert json.loads(next(handle))["model_identity"] == {"revision": "before"}


def test_qwen_loaded_metadata_wrapper_and_no_dtype_traversal():
    from transformers import Qwen2_5_VLConfig

    from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL

    root = Qwen2_5_VLConfig()
    root._commit_hash = "a" * 40
    root._attn_implementation = "sdpa"
    root.text_config._commit_hash = "b" * 40
    root.text_config._attn_implementation = "eager"

    class Loaded:
        is_quantized = False
        hf_quantizer = None

        @property
        def dtype(self):
            raise AssertionError("model.dtype traverses parameters")

    loaded = Loaded()
    loaded.config = root
    loaded.get_input_embeddings = lambda: SimpleNamespace(weight=torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16)))
    loaded.parameters = Mock(side_effect=AssertionError("parameter traversal"))
    adapter = Qwen2_5_VL.__new__(Qwen2_5_VL)
    adapter._model = object()
    adapter.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: loaded)
    identity = adapter.response_cache_identity()
    assert identity["input_embedding_dtype"] == "torch.bfloat16"
    assert identity["root_config"]["commit_hash"] == "a" * 40
    assert identity["text_config"]["attention"] == "eager"
    assert identity["vision_config"]["commit_hash"] is None
    loaded.get_input_embeddings = lambda: SimpleNamespace(weight=torch.nn.Parameter(torch.empty(1, dtype=torch.int8), requires_grad=False))
    with pytest.raises(ValueError, match="dtype"):
        adapter.response_cache_identity()
    loaded.get_input_embeddings = lambda: SimpleNamespace(weight=torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16)))
    root.vision_config._commit_hash = 3
    with pytest.raises(ValueError, match="commit hash"):
        adapter.response_cache_identity()
    root.vision_config._commit_hash = None
    root.text_config._attn_implementation = ""
    with pytest.raises(ValueError, match="attention"):
        adapter.response_cache_identity()


@pytest.mark.parametrize("quantized_source", ["is_quantized", "hf_quantizer", "root", "text", "vision"])
def test_quantized_qwen_preserves_legacy_cache_keys(tmp_path, quantized_source):
    from transformers import Qwen2_5_VLConfig

    from lmms_eval.models.simple import qwen2_5_vl

    root = Qwen2_5_VLConfig()
    loaded = SimpleNamespace(config=root, is_quantized=False, hf_quantizer=None)
    loaded.get_input_embeddings = Mock(side_effect=AssertionError("quantized identity must not inspect embeddings"))
    if quantized_source == "is_quantized":
        loaded.is_quantized = True
    elif quantized_source == "hf_quantizer":
        loaded.hf_quantizer = object()
    else:
        part = root if quantized_source == "root" else getattr(root, f"{quantized_source}_config")
        part.quantization_config = {"method": "bitsandbytes"}

    adapter = qwen2_5_vl.Qwen2_5_VL.__new__(qwen2_5_vl.Qwen2_5_VL)
    adapter._model = loaded
    adapter.generate_until = Mock(side_effect=AssertionError("legacy cache should hit"))
    cache_root = tmp_path / "cache"
    assert run(cache_root, Legacy("legacy"))[0] == ["legacy"]
    with patch.object(qwen2_5_vl, "eval_logger") as logger:
        print("PR1623_QUANTIZED_CACHE_SEAM_READY")
        result, stats = run(cache_root, adapter)
    assert result == ["legacy"]
    assert stats["hits"] == 1
    adapter.generate_until.assert_not_called()
    loaded.get_input_embeddings.assert_not_called()
    logger.warning.assert_called_once()
    assert "legacy cache key" in logger.warning.call_args.args[0]
    with sqlite3.connect(cache_root / "cache.db") as db:
        assert json.loads(db.execute("SELECT value FROM meta WHERE key='model_identity'").fetchone()[0]) is None
        assert "effective_identity_v1" not in db.execute("SELECT value FROM meta WHERE key='model_fingerprint'").fetchone()[0]
