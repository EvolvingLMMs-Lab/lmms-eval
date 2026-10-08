import inspect
import unittest
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from lmms_eval.models.model_utils.grt import qwen2_5_vl as qwen25_module
from lmms_eval.models.model_utils.grt.qwen2_5_vl import (
    GatedQwenVisionPatchEmbed,
    Qwen2_5_VL,
    _count_feature_tokens,
    _count_grid_tokens,
    _read_video_metadata_pyav,
)


class _TinyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(1, 2, kernel_size=(1, 1, 1), bias=True)
        self.patch_size = 1
        self.temporal_patch_size = 1
        self.in_channels = 1
        self.embed_dim = 2


class Qwen25GRTMetricsTest(unittest.TestCase):
    def test_revision_is_explicit_and_forwarded_to_all_hf_loaders(self):
        signature = inspect.signature(Qwen2_5_VL.__init__)
        self.assertEqual(signature.parameters["revision"].default, "main")
        source = inspect.getsource(Qwen2_5_VL.__init__)
        self.assertGreaterEqual(source.count("revision=revision"), 4)

    def test_throughput_timer_covers_visual_loading_and_preprocessing(self):
        source = inspect.getsource(Qwen2_5_VL.generate_until)

        timer = source.index("run_wall_start = time.perf_counter()")
        visual_loading = source.index("doc_to_visual[0]")
        preprocessing = source.index("process_vision_info(messages)")
        generation = source.index("self.model.generate(")
        decoding = source.index("self.processor.batch_decode(")
        elapsed = source.index("time.perf_counter() - run_wall_start")

        self.assertLess(timer, visual_loading)
        self.assertLess(timer, preprocessing)
        self.assertLess(timer, generation)
        self.assertGreater(elapsed, decoding)

    def test_custom_loader_uses_pyav_metadata_and_preserves_eight_frame_input(self):
        stream = SimpleNamespace(
            average_rate=Fraction(30, 1),
            frames=240,
            duration=None,
            time_base=None,
        )

        class FakeContainer:
            streams = SimpleNamespace(video=[stream])

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        model = Qwen2_5_VL.__new__(Qwen2_5_VL)
        model.max_num_frames = 8
        model.fps = None
        model.max_image_size = None
        model.profiling = True
        encoded_frames = [f"frame-{index}" for index in range(8)]
        stream_output = StringIO()

        with (
            mock.patch.object(qwen25_module.av, "open", return_value=FakeContainer()) as av_open,
            mock.patch.object(
                qwen25_module,
                "read_video_pyav_seek_base64",
                return_value=encoded_frames,
            ) as pyav_sampler,
            mock.patch.object(
                qwen25_module,
                "decord",
                SimpleNamespace(VideoReader=mock.Mock(side_effect=AssertionError("custom PyAV loading must not instantiate Decord"))),
            ),
            redirect_stdout(stream_output),
        ):
            actual = model._load_video_base64("fixture.mp4")

        self.assertIs(actual, encoded_frames)
        av_open.assert_called_once_with("fixture.mp4")
        pyav_sampler.assert_called_once_with(
            "fixture.mp4",
            num_frm=8,
            fps=None,
            img_format="JPEG",
            max_image_size=None,
        )
        self.assertEqual(model._last_sampled_frames, 8)
        self.assertEqual(model._last_requested_frames, 8)
        self.assertEqual(model._last_reference_frames, 8)
        self.assertAlmostEqual(model._last_effective_fps, 1.0)
        self.assertIn(
            "orig_fps=30.000000 duration_s=8.000000 total_frames=240 capped_frames=240 sampled_frames=8 effective_fps=1.000000",
            stream_output.getvalue(),
        )

    def test_pyav_metadata_uses_stream_duration_when_frame_count_is_missing(self):
        stream = SimpleNamespace(
            average_rate=Fraction(25, 1),
            frames=0,
            duration=400,
            time_base=Fraction(1, 100),
        )

        class FakeContainer:
            streams = SimpleNamespace(video=[stream])

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        with mock.patch.object(qwen25_module.av, "open", return_value=FakeContainer()):
            total_frames, orig_fps, duration_sec = _read_video_metadata_pyav("fixture.mp4")

        self.assertEqual(total_frames, 100)
        self.assertEqual(orig_fps, 25.0)
        self.assertEqual(duration_sec, 4.0)

    def test_counts_tuple_returned_by_get_video_features(self):
        outputs = (torch.empty(4, 8), torch.empty(7, 8))

        self.assertEqual(_count_feature_tokens(outputs), 11)

    def test_counts_legacy_pooler_output(self):
        outputs = SimpleNamespace(pooler_output=(torch.empty(3, 8), torch.empty(5, 8)))

        self.assertEqual(_count_feature_tokens(outputs), 8)

    def test_counts_grid_patch_tubes(self):
        grid = torch.tensor([[2, 3, 4], [1, 2, 5]])

        self.assertEqual(_count_grid_tokens(grid), 34)

    def test_all_policy_is_exact_native_conv_control(self):
        original = _TinyPatchEmbed()
        parent = SimpleNamespace(
            _last_reference_orig_patches=3,
            profiling=False,
        )
        gated = GatedQwenVisionPatchEmbed(
            original,
            diff_threshold=0.3,
            gate_policy="all",
            parent=parent,
        )
        gated.set_grid_thw(torch.tensor([[3, 1, 1]]), is_video=True)
        hidden = torch.tensor([[0.0], [0.3], [0.6]])

        expected = original.proj(hidden.view(-1, 1, 1, 1, 1)).view(-1, 2)
        actual = gated(hidden)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(parent._last_recomputed_patches, 3)
        self.assertEqual(parent._last_orig_patches, 3)
        self.assertEqual(parent._last_patch_projection_recompute_ratio, 1.0)
        self.assertEqual(parent._last_patch_projection_compute_ratio_vs_reference, 1.0)

    def test_motion_compares_against_last_recomputed_patch(self):
        original = _TinyPatchEmbed()
        with torch.no_grad():
            original.proj.weight.fill_(1.0)
            original.proj.bias.zero_()
        parent = SimpleNamespace(
            _last_reference_orig_patches=3,
            profiling=False,
        )
        gated = GatedQwenVisionPatchEmbed(
            original,
            diff_threshold=0.5,
            gate_policy="motion",
            parent=parent,
        )
        gated.set_grid_thw(torch.tensor([[3, 1, 1]]), is_video=True)
        hidden = torch.tensor([[0.0], [0.3], [0.6]])

        actual = gated(hidden)

        self.assertEqual(gated.last_keep_flat.tolist(), [True, False, True])
        self.assertEqual(parent._last_recomputed_patches, 2)
        self.assertAlmostEqual(parent._last_patch_projection_recompute_ratio, 2 / 3)
        self.assertTrue(torch.equal(actual[:, 0], torch.tensor([0.0, 0.0, 0.6])))

    def test_dense_metrics_exposes_reference_scope_for_base(self):
        model = Qwen2_5_VL.__new__(Qwen2_5_VL)
        model.profiling = True
        model._last_sampled_frames = 8
        model._last_requested_frames = 8
        model._last_reference_frames = 8
        model._last_effective_fps = 0.25
        model._last_pre_tokens = 12
        model._last_post_tokens = 12
        model._last_gate_keep_ratio = 1.0
        model._last_recomputed_patches = 12
        model._last_orig_patches = 12
        model._last_recompute_ratio = 1.0
        model._last_gate_policy = "disabled"
        model._last_gate_metric = "none"
        model._last_reference_orig_patches = 12
        model._last_patch_projection_recompute_ratio = 1.0
        model._last_patch_projection_compute_ratio_vs_reference = 1.0
        model._last_tokenization_time = 0.1
        stream = StringIO()

        with redirect_stdout(stream):
            model._print_dense_metrics(2.0)

        output = stream.getvalue()
        self.assertIn("requested_frames=8", output)
        self.assertIn("reference_frames=8", output)
        self.assertIn("recomputed_patches=12", output)
        self.assertIn("orig_patches=12", output)
        self.assertIn("reference_orig_patches=12", output)
        self.assertIn("patch_projection_recompute_ratio=1.000000", output)
        self.assertIn("patch_projection_compute_ratio_vs_reference=1.000000", output)


if __name__ == "__main__":
    unittest.main()
