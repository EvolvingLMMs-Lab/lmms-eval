"""Opt-in real filesystem / GPU smoke test.

LMMS_CACHE_E2E_ROOT=/mnt/nfs/cache-tests LMMS_CACHE_E2E_DEVICE=cuda \
    python -m pytest test/cache/test_response_cache_integration.py -q -s

Uses a tiny randomly initialized GPT-2 for offline deterministic generation;
no model download or benchmark dataset is needed. Supply an actual NFS mount
as the root to exercise NFS publication rather than a simulated filesystem.
"""

import os
import tempfile

import pytest

from lmms_eval.api.instance import Instance
from lmms_eval.caching.fs_detect import detect_fs_type
from lmms_eval.caching.response_cache import ResponseCache


@pytest.mark.skipif(not os.environ.get("LMMS_CACHE_E2E_ROOT"), reason="set LMMS_CACHE_E2E_ROOT to enable filesystem integration")
def test_generation_publication_and_replay(monkeypatch):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    device = os.environ.get("LMMS_CACHE_E2E_DEVICE", "cpu")
    torch.manual_seed(1614)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=32, n_layer=1, n_head=2, bos_token_id=1, eos_token_id=31, pad_token_id=0)).eval().to(device)

    class Adapter:
        calls = 0

        def generate_until(self, requests):
            self.calls += len(requests)
            responses = []
            for request in requests:
                ids = torch.tensor([[1, 2, request.doc_id + 3]], device=device)
                with torch.inference_mode():
                    generated = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=4)
                responses.append(" ".join(str(token) for token in generated[0].tolist()))
            return responses

    requests = [Instance(request_type="generate_until", arguments=(f"prompt-{index}", {"temperature": 0}), idx=0, metadata={"task": "snapshot_e2e", "doc_id": index, "repeats": 1}) for index in range(4)]
    adapter = Adapter()
    root = os.environ["LMMS_CACHE_E2E_ROOT"]
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="response-cache-e2e-", dir=root) as cache_root:

        def create(run):
            monkeypatch.setenv("LMMS_CACHE_RUN_ID", run)
            return ResponseCache.create(cache_root, model="tiny-gpt2-snapshot-test")

        first = create("first")
        try:
            expected = first.execute(adapter, "generate_until", requests)
            assert adapter.calls == 4
            first.finalize(success=True)
        finally:
            first.close()
        second = create("second")
        try:
            assert second.execute(adapter, "generate_until", requests) == expected
            stats = second.get_stats()
            assert stats["hits"] == stats["hits_shared"] == 4
            assert stats["misses"] == 0
            assert adapter.calls == 4
            second.finalize(success=True)
        finally:
            second.close()
        assert not os.path.exists(os.path.join(cache_root, "cache.db-wal"))
        assert not os.path.exists(os.path.join(cache_root, "cache.db-shm"))
        print({"device": device, "filesystem": detect_fs_type(cache_root).value, "model_calls": adapter.calls, "replay_shared_hits": stats["hits_shared"]})
