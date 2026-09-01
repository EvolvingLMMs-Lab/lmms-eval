import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from lmms_eval.models import get_model, get_model_manifest
from lmms_eval.models.chat.sglang import Sglang
from lmms_eval.models.chat.sglang_diffusion import (
    SGLangDiffusion,
    _infer_output_type,
    is_sglang_diffusion_model,
)


class _FakeAccelerator:
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")

    def free_memory(self):
        pass


class _FakeDiffusionClient:
    def __init__(self):
        self.sampling_params = []
        self.shutdown_called = False

    def generate(self, sampling_params_kwargs):
        self.sampling_params.append(sampling_params_kwargs)
        output_path = os.path.join(sampling_params_kwargs["output_path"], sampling_params_kwargs["output_file_name"])
        Path(output_path).write_bytes(b"fake mp4")
        return SimpleNamespace(output_file_path=output_path)

    def shutdown(self):
        self.shutdown_called = True


class _FakeDiffGenerator:
    last_local_mode = None
    last_server_kwargs = None
    last_client = None

    @classmethod
    def from_pretrained(cls, local_mode=True, **kwargs):
        cls.last_local_mode = local_mode
        cls.last_server_kwargs = kwargs
        cls.last_client = _FakeDiffusionClient()
        return cls.last_client


class TestSGLangDiffusionDetection(unittest.TestCase):
    def test_wan_diffusers_repo_uses_diffusion_runtime(self):
        self.assertTrue(is_sglang_diffusion_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers"))
        self.assertFalse(is_sglang_diffusion_model("Qwen/Qwen3-VL-8B-Instruct"))
        self.assertEqual(_infer_output_type("Wan-AI/Wan2.2-I2V-A14B-Diffusers"), "video")
        self.assertEqual(_infer_output_type("Qwen/Qwen-Image"), "image")

    def test_runtime_override_takes_precedence(self):
        self.assertTrue(is_sglang_diffusion_model("custom/checkpoint", runtime="diffusion"))
        self.assertFalse(is_sglang_diffusion_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers", runtime="language"))

    def test_local_diffusers_model_index_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model_index.json").write_text(
                json.dumps({"_class_name": "WanImageToVideoPipeline", "scheduler": ["diffusers", "Scheduler"], "transformer": ["diffusers", "WanTransformer"]}),
                encoding="utf-8",
            )
            self.assertTrue(is_sglang_diffusion_model(tmpdir))

    def test_registry_exposes_explicit_diffusion_aliases(self):
        manifest = get_model_manifest("sglang-wan")
        self.assertEqual(manifest.model_id, "sglang_diffusion")
        self.assertIs(get_model("sglang-diffusion"), SGLangDiffusion)


class TestSGLangDiffusionGeneration(unittest.TestCase):
    def test_standard_sglang_entrypoint_routes_wan_to_diffusion(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch("lmms_eval.models.chat.sglang_diffusion.Accelerator", _FakeAccelerator), patch("lmms_eval.models.chat.sglang_diffusion._load_diff_generator", return_value=_FakeDiffGenerator):
            model = Sglang(
                model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                output_dir=tmpdir,
                num_gpus=4,
            )
            self.assertIsInstance(model, SGLangDiffusion)
            self.assertEqual(model.num_gpus, 4)
            model.close()

    def test_missing_runtime_result_does_not_reuse_stale_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stale_path = Path(tmpdir, "video.mp4")
            stale_path.write_bytes(b"stale")
            self.assertEqual(SGLangDiffusion._result_paths(None), [])

    def test_wan_request_generates_vbvr_video_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch("lmms_eval.models.chat.sglang_diffusion.Accelerator", _FakeAccelerator), patch("lmms_eval.models.chat.sglang_diffusion._load_diff_generator", return_value=_FakeDiffGenerator):
            model = SGLangDiffusion(
                model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                output_dir=tmpdir,
                num_gpus=4,
                enable_cfg_parallel=True,
                ulysses_degree=2,
                text_encoder_cpu_offload=True,
            )
            image = Image.new("RGB", (64, 48), color=(20, 40, 60))
            doc = {"prompt": "Move the red square to the right", "image": image}
            model.task_dict = {"vbvr": {"test": [doc]}}

            def doc_to_messages(current_doc):
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": current_doc["image"]},
                            {"type": "text", "text": current_doc["prompt"]},
                        ],
                    }
                ]

            request = SimpleNamespace(arguments=(None, doc_to_messages, {"num_frames": 33, "max_new_tokens": 1}, 0, "vbvr", "test"))
            responses = model.generate_until([request])

            payload = json.loads(responses[0].text)
            self.assertEqual(payload["text"], "")
            self.assertEqual(len(payload["videos"]), 1)
            self.assertTrue(os.path.isfile(payload["videos"][0]))

            self.assertTrue(os.path.isfile(_FakeDiffGenerator.last_client.sampling_params[0]["image_path"]))
            self.assertEqual(_FakeDiffGenerator.last_client.sampling_params[0]["prompt"], doc["prompt"])
            self.assertEqual(_FakeDiffGenerator.last_client.sampling_params[0]["num_frames"], 33)
            self.assertNotIn("max_new_tokens", _FakeDiffGenerator.last_client.sampling_params[0])
            self.assertEqual(_FakeDiffGenerator.last_server_kwargs["model_path"], "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            self.assertEqual(_FakeDiffGenerator.last_server_kwargs["num_gpus"], 4)
            self.assertTrue(_FakeDiffGenerator.last_server_kwargs["enable_cfg_parallel"])
            self.assertEqual(_FakeDiffGenerator.last_server_kwargs["ulysses_degree"], 2)

            model.close()
            self.assertTrue(_FakeDiffGenerator.last_client.shutdown_called)
            model.close()


if __name__ == "__main__":
    unittest.main()
