import os
import tempfile
import unittest
from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

from lmms_eval.api.instance import GenerationResult
from lmms_eval.models.chat.fastvideo import FastVideo


class _FakeVideoGenerator:
    last_model = None
    last_kwargs = None

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.last_model = model_path
        cls.last_kwargs = kwargs
        return object()


class TestFastVideoGenerateUntil(unittest.TestCase):
    def test_init_signature_does_not_expose_overwrite(self):
        self.assertNotIn("overwrite", signature(FastVideo.__init__).parameters)

    def test_init_does_not_forward_legacy_overwrite_kwarg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lmms_eval.models.chat.fastvideo._has_fastvideo", True), patch("lmms_eval.models.chat.fastvideo.VideoGenerator", _FakeVideoGenerator):
                FastVideo(model="demo-model", output_dir=tmpdir, overwrite=True)

        self.assertEqual(_FakeVideoGenerator.last_model, "demo-model")
        self.assertNotIn("overwrite", _FakeVideoGenerator.last_kwargs)

    def test_generate_until_does_not_reuse_existing_output_file_internally(self):
        model = FastVideo.__new__(FastVideo)
        model.data_parallel = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "existing.mp4")
            with open(output_path, "wb") as handle:
                handle.write(b"existing")

            prepared = {"prompt": "demo", "image_path": "/tmp/input.png", "output_path": output_path}
            expected = [GenerationResult(text='{"text":"","videos":["/tmp/generated.mp4"]}')]

            model.make_one_request = lambda request: prepared
            calls = []

            def _fake_generate_until_single(items):
                calls.append(items)
                return expected

            model._generate_until_single = _fake_generate_until_single
            model._generate_until_parallel = lambda items: self.fail("parallel path should not be used")

            result = model.generate_until([SimpleNamespace()])

        self.assertEqual(calls, [[prepared]])
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
