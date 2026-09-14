import copy
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

from lmms_eval.models.model_utils.grt.llava_hf import (
    DENSEVIDEO_LPM_PROMPT_ROUTER,
    GATE_PROJECTION_LINEAR_CONSISTENT,
    GATE_PROJECTION_NATIVE_ON_FULL,
    GatedSiglipVisionEmbeddings,
    LlavaHf,
    classify_densevideo_lpm_first_sentence,
    resolve_video_request_plan,
)


class LlavaHfGatedEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = SiglipVisionConfig(
            hidden_size=8,
            image_size=8,
            patch_size=4,
            num_channels=3,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        self.native = SiglipVisionEmbeddings(config).eval()

    def _gated(
        self,
        *,
        policy="motion",
        threshold=0.01,
        metric="ssim",
        projection_mode=GATE_PROJECTION_NATIVE_ON_FULL,
        refresh_interval=0,
    ):
        parent = SimpleNamespace()
        module = GatedSiglipVisionEmbeddings(
            copy.deepcopy(self.native),
            diff_threshold=threshold,
            gate_policy=policy,
            gate_metric=metric,
            gate_projection_mode=projection_mode,
            gate_refresh_interval_frames=refresh_interval,
            parent=parent,
        ).eval()
        return module, parent

    def test_gate_policy_all_is_exact_native_conv_and_position_embedding(self):
        pixel_values = torch.randn(4, 3, 8, 8)
        gated, parent = self._gated(policy="all")
        gated.set_video_frames(4)

        expected = self.native(pixel_values)
        actual = gated(pixel_values)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(parent._last_gate_policy, "all")
        self.assertEqual(parent._last_recomputed_patches, parent._last_orig_patches)
        self.assertEqual(parent._last_recompute_ratio, 1.0)

    def test_repeated_frames_reuse_patch_embeddings(self):
        frame = torch.randn(1, 3, 8, 8)
        pixel_values = frame.repeat(4, 1, 1, 1)
        gated, parent = self._gated(policy="motion", threshold=0.01)
        gated.set_video_frames(4)

        expected = self.native(pixel_values)
        actual = gated(pixel_values)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(parent._last_gate_policy, "motion")
        self.assertEqual(parent._last_gate_metric, "ssim")
        self.assertEqual(parent._last_recomputed_patches, self.native.num_patches)
        self.assertEqual(parent._last_orig_patches, 4 * self.native.num_patches)
        self.assertLess(parent._last_recompute_ratio, 1.0)
        self.assertEqual(parent._last_recompute_ratio, 0.25)

    def test_cumulative_drift_is_measured_from_cached_embedding_source(self):
        base = torch.zeros(1, 3, 8, 8)
        # Each adjacent L2 patch delta is below 0.5, while the second frame's
        # cumulative delta from the cached base patch is above 0.5.
        pixel_values = torch.cat([base, base + 0.05, base + 0.10, base + 0.15], dim=0)
        gated, parent = self._gated(policy="motion", threshold=0.5, metric="l2")
        gated.set_video_frames(4)

        gated(pixel_values)

        self.assertEqual(parent._last_recomputed_patches, 2 * self.native.num_patches)
        self.assertEqual(parent._last_orig_patches, 4 * self.native.num_patches)
        self.assertEqual(parent._last_recompute_ratio, 0.5)

    def test_request_local_policy_does_not_leak_to_the_next_video(self):
        frame = torch.randn(1, 3, 8, 8)
        pixel_values = frame.repeat(4, 1, 1, 1)
        gated, parent = self._gated(policy="motion", threshold=0.01)

        gated.set_video_context(4, gate_policy="all")
        exact = gated(pixel_values)
        torch.testing.assert_close(exact, self.native(pixel_values), rtol=0, atol=0)
        self.assertEqual(parent._last_gate_policy, "all")
        self.assertEqual(parent._last_recompute_ratio, 1.0)

        # set_video_frames intentionally clears the request-local override and
        # restores the configured motion policy.
        gated.set_video_frames(4)
        reused = gated(pixel_values)
        torch.testing.assert_close(reused, self.native(pixel_values), rtol=0, atol=0)
        self.assertEqual(parent._last_gate_policy, "motion")
        self.assertEqual(parent._last_recompute_ratio, 0.25)

        gated.set_video_context(None)
        self.assertIsNone(gated._active_video_frames)
        self.assertIsNone(gated._active_gate_policy)

    def test_motion_all_keep_frames_use_exact_native_conv(self):
        frames = [torch.zeros(1, 3, 8, 8)]
        frames.extend(torch.full((1, 3, 8, 8), float(value)) for value in (1, 2, 3))
        pixel_values = torch.cat(frames, dim=0)
        gated, parent = self._gated(policy="motion", threshold=0.0, metric="l2")
        gated.set_video_frames(4)
        conv_calls = []
        hook = gated.patch_embedding.register_forward_hook(lambda *_: conv_calls.append(True))

        try:
            actual = gated(pixel_values)
        finally:
            hook.remove()

        torch.testing.assert_close(actual, self.native(pixel_values), rtol=0, atol=0)
        self.assertEqual(len(conv_calls), 4)
        self.assertEqual(parent._last_recomputed_patches, parent._last_orig_patches)
        self.assertEqual(parent._last_recompute_ratio, 1.0)
        self.assertEqual(parent._last_gate_projection_mode, GATE_PROJECTION_NATIVE_ON_FULL)

    def test_linear_consistent_reproduces_old_all_keep_kernel_and_calls_conv_once(self):
        frames = [torch.zeros(1, 3, 8, 8)]
        frames.extend(torch.full((1, 3, 8, 8), float(value)) for value in (1, 2, 3))
        pixel_values = torch.cat(frames, dim=0)
        gated, parent = self._gated(
            policy="motion",
            threshold=0.0,
            metric="l2",
            projection_mode=GATE_PROJECTION_LINEAR_CONSISTENT,
        )
        gated.set_video_frames(4)

        # This is the pre-native-on-full kernel path: native Conv2d for the
        # first frame, then the same flattened weight through F.linear for all
        # selected patches in every later frame.
        patches = torch.nn.functional.unfold(
            pixel_values,
            kernel_size=self.native.patch_embedding.kernel_size,
            dilation=self.native.patch_embedding.dilation,
            padding=(0 if self.native.patch_embedding.padding == "valid" else self.native.patch_embedding.padding),
            stride=self.native.patch_embedding.stride,
        ).transpose(1, 2)
        first = self.native.patch_embedding(pixel_values[:1]).flatten(2).transpose(1, 2)
        later = torch.nn.functional.linear(
            patches[1:],
            self.native.patch_embedding.weight.flatten(1),
            self.native.patch_embedding.bias,
        )
        expected = torch.cat((first, later), dim=0)
        expected = expected + self.native.position_embedding(self.native.position_ids)

        conv_calls = []
        hook = gated.patch_embedding.register_forward_hook(lambda *_: conv_calls.append(True))
        try:
            actual = gated(pixel_values)
        finally:
            hook.remove()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(conv_calls), 1)
        self.assertEqual(parent._last_gate_projection_mode, GATE_PROJECTION_LINEAR_CONSISTENT)
        self.assertEqual(parent._last_forced_refresh_frames, 0)
        self.assertEqual(parent._last_forced_refresh_patches, 0)

    def test_refresh_one_is_exact_native_and_records_every_later_frame(self):
        pixel_values = torch.randn(4, 3, 8, 8)
        gated, parent = self._gated(
            policy="motion",
            threshold=1e9,
            metric="l2",
            refresh_interval=1,
        )
        gated.set_video_frames(4)

        actual = gated(pixel_values)

        torch.testing.assert_close(actual, self.native(pixel_values), rtol=0, atol=0)
        self.assertEqual(parent._last_recompute_ratio, 1.0)
        self.assertEqual(parent._last_forced_refresh_frames, 3)
        self.assertEqual(
            parent._last_forced_refresh_patches,
            3 * self.native.num_patches,
        )
        self.assertEqual(parent._last_forced_refresh_frame_indices, "1,2,3")

    def test_refresh_three_forces_zero_indexed_frames_three_and_six(self):
        frame = torch.randn(1, 3, 8, 8)
        pixel_values = frame.repeat(8, 1, 1, 1)
        gated, parent = self._gated(
            policy="motion",
            threshold=0.01,
            refresh_interval=3,
        )
        gated.set_video_frames(8)

        actual = gated(pixel_values)

        torch.testing.assert_close(actual, self.native(pixel_values), rtol=0, atol=0)
        expected_full = {0, 3, 6}
        for frame_index, frame_keep in enumerate(gated.last_keep_flat):
            self.assertEqual(bool(frame_keep.all()), frame_index in expected_full)
        self.assertEqual(parent._last_recomputed_patches, 3 * self.native.num_patches)
        self.assertEqual(parent._last_orig_patches, 8 * self.native.num_patches)
        self.assertEqual(parent._last_recompute_ratio, 3 / 8)
        self.assertEqual(parent._last_gate_refresh_interval_frames, 3)
        self.assertEqual(parent._last_forced_refresh_frames, 2)
        self.assertEqual(parent._last_forced_refresh_frame_indices, "3,6")

    def test_invalid_projection_or_refresh_arguments_fail_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "gate_projection_mode"):
            self._gated(projection_mode="implicit_legacy")
        for value in (-1, 1.5, True):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError,
                    "gate_refresh_interval_frames",
                ),
            ):
                self._gated(refresh_interval=value)

        with mock.patch("lmms_eval.models.model_utils.grt.llava_hf.AutoConfig.from_pretrained") as load_config, mock.patch("lmms_eval.models.model_utils.grt.llava_hf.Accelerator") as load_accelerator:
            with self.assertRaisesRegex(ValueError, "gate_projection_mode"):
                LlavaHf(gate_projection_mode="implicit_legacy")
            load_config.assert_not_called()
            load_accelerator.assert_not_called()

    def test_reference_patch_budget_uses_eight_frame_control(self):
        pixel_values = torch.randn(12, 3, 8, 8)
        gated, parent = self._gated(policy="all")
        parent._last_reference_frames = 8
        gated.set_video_frames(12)

        gated(pixel_values)

        expected_reference = 8 * self.native.num_patches
        self.assertEqual(parent._last_reference_orig_patches, expected_reference)
        self.assertEqual(parent._last_patch_projection_recompute_ratio, 1.0)
        self.assertEqual(parent._last_patch_projection_compute_ratio_vs_reference, 1.5)


class LlavaHfPromptRouterTest(unittest.TestCase):
    def test_classifier_uses_only_the_first_sentence(self):
        self.assertEqual(
            classify_densevideo_lpm_first_sentence("  What   subtitles appear in the entire video clip-1? What text is extracted by OCR later?"),
            "subtitle",
        )
        self.assertEqual(
            classify_densevideo_lpm_first_sentence("WHAT TEXT IS EXTRACTED BY OCR IN THE ENTIRE VIDEO clip-1? Mention subtitles too."),
            "ocr",
        )
        self.assertEqual(
            classify_densevideo_lpm_first_sentence("Describe the video. What text is extracted by OCR in the entire video?"),
            "unknown",
        )

    def test_query_adaptive_plan_is_exact_for_subtitle_and_unknown(self):
        common = {
            "prompt_router": DENSEVIDEO_LPM_PROMPT_ROUTER,
            "max_frames_num": 32,
            "ocr_max_frames_num": 12,
            "gate_policy": "motion",
            "video_decode_backend": "pyav_seek",
            "default_max_new_tokens": 128,
            "subtitle_video_decode_backend": "decord_frozen",
            "subtitle_max_new_tokens": 31,
        }
        subtitle = resolve_video_request_plan(
            "What subtitles appear in the entire video clip? More instructions.",
            **common,
        )
        ocr = resolve_video_request_plan(
            "What text is extracted by OCR in the entire video clip? More instructions.",
            **common,
        )
        unknown = resolve_video_request_plan("Describe this clip. OCR may be present.", **common)

        self.assertEqual((subtitle.question_route, subtitle.requested_frames, subtitle.gate_policy), ("subtitle", 8, "all"))
        self.assertEqual((ocr.question_route, ocr.requested_frames, ocr.gate_policy), ("ocr", 12, "motion"))
        self.assertEqual((unknown.question_route, unknown.requested_frames, unknown.gate_policy), ("unknown", 8, "all"))
        self.assertEqual(subtitle.reference_frames, 8)
        self.assertEqual(ocr.reference_frames, 8)
        self.assertEqual(
            (subtitle.video_decode_backend, subtitle.max_new_tokens),
            ("decord_frozen", 31),
        )
        self.assertEqual(
            (ocr.video_decode_backend, ocr.max_new_tokens),
            ("pyav_seek", 128),
        )
        self.assertEqual(
            (unknown.video_decode_backend, unknown.max_new_tokens),
            ("pyav_seek", 128),
        )

    def test_router_off_preserves_configured_behavior(self):
        plan = resolve_video_request_plan(
            "What subtitles appear in the entire video clip?",
            prompt_router="off",
            max_frames_num=16,
            ocr_max_frames_num=12,
            gate_policy="motion",
        )

        self.assertEqual(plan.question_route, "unrouted")
        self.assertEqual(plan.requested_frames, 16)
        self.assertEqual(plan.reference_frames, 16)
        self.assertEqual(plan.gate_policy, "motion")
        self.assertEqual(plan.video_decode_backend, "pyav_seek")
        self.assertEqual(plan.max_new_tokens, 1024)

    def test_route31_changes_only_subtitle_token_cap_on_pyav(self):
        common = {
            "prompt_router": DENSEVIDEO_LPM_PROMPT_ROUTER,
            "max_frames_num": 8,
            "ocr_max_frames_num": 8,
            "gate_policy": "motion",
            "video_decode_backend": "pyav_seek",
            "default_max_new_tokens": 128,
            "subtitle_max_new_tokens": 31,
        }
        subtitle = resolve_video_request_plan("What subtitles appear in the entire video clip?", **common)
        ocr = resolve_video_request_plan("What text is extracted by OCR in the entire video clip?", **common)

        self.assertEqual(
            (subtitle.video_decode_backend, subtitle.requested_frames, subtitle.max_new_tokens),
            ("pyav_seek", 8, 31),
        )
        self.assertEqual(
            (ocr.video_decode_backend, ocr.requested_frames, ocr.max_new_tokens),
            ("pyav_seek", 8, 128),
        )

    def test_invalid_router_or_frame_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported prompt_router"):
            resolve_video_request_plan(
                "question",
                prompt_router="hidden_label",
                max_frames_num=8,
                ocr_max_frames_num=12,
                gate_policy="motion",
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_video_request_plan(
                "question",
                prompt_router=DENSEVIDEO_LPM_PROMPT_ROUTER,
                max_frames_num=8,
                ocr_max_frames_num=0,
                gate_policy="motion",
            )
        with self.assertRaisesRegex(ValueError, "subtitle_max_new_tokens requires"):
            resolve_video_request_plan(
                "question",
                prompt_router="off",
                max_frames_num=8,
                ocr_max_frames_num=8,
                gate_policy="motion",
                subtitle_max_new_tokens=32,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported video_decode_backend"):
            resolve_video_request_plan(
                "question",
                prompt_router=DENSEVIDEO_LPM_PROMPT_ROUTER,
                max_frames_num=8,
                ocr_max_frames_num=8,
                gate_policy="motion",
                video_decode_backend="random_decoder",
            )


class LlavaHfVideoDecodeBackendTest(unittest.TestCase):
    @staticmethod
    def _bare_model(backend):
        model = object.__new__(LlavaHf)
        model.video_decode_backend = backend
        model.max_image_size = None
        model.profiling = False
        model._last_reference_frames = 8
        model._last_requested_frames = 10
        model._last_prompt_router = DENSEVIDEO_LPM_PROMPT_ROUTER
        model._last_question_route = "ocr"
        return model

    def test_pyav_seek_caps_short_video_and_reference(self):
        stream = SimpleNamespace(average_rate=25.0, frames=6, duration=None, time_base=None)
        container = mock.MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [stream]
        decoded = np.zeros((6, 4, 4, 3), dtype=np.uint8)
        model = self._bare_model("pyav_seek")

        with (
            mock.patch("lmms_eval.models.model_utils.grt.llava_hf.av.open", return_value=container),
            mock.patch(
                "lmms_eval.models.model_utils.grt.llava_hf.read_video_pyav_seek_uniform",
                return_value=decoded,
            ) as read_video,
        ):
            actual = model.load_video("clip.mp4", 10)

        self.assertEqual(actual.shape[0], 6)
        self.assertEqual(read_video.call_args.kwargs["num_frm"], 6)
        self.assertEqual(model._last_sampled_frames, 6)
        self.assertEqual(model._last_reference_frames, 6)

    def test_decord_legacy_preserves_requested_count_and_linspace_indices(self):
        captured = {}

        class Batch:
            def __init__(self, count):
                self.count = count

            def asnumpy(self):
                return np.zeros((self.count, 4, 4, 3), dtype=np.uint8)

        class FakeVideoReader:
            def __init__(self, path, ctx):
                captured["path"] = path
                captured["ctx"] = ctx

            def __len__(self):
                return 6

            def get_avg_fps(self):
                return 25.0

            def get_batch(self, indices):
                captured["indices"] = indices
                return Batch(len(indices))

        fake_decord = SimpleNamespace(VideoReader=FakeVideoReader, cpu=lambda index: ("cpu", index))
        model = self._bare_model("decord_legacy")

        with mock.patch.dict(sys.modules, {"decord": fake_decord}):
            actual = model.load_video("clip.mp4", 10)

        self.assertEqual(actual.shape[0], 10)
        self.assertEqual(captured["indices"], np.linspace(0, 5, 10, dtype=int).tolist())
        self.assertEqual(model._last_sampled_frames, 10)
        self.assertEqual(model._last_reference_frames, 8)

    def test_frozen_decord_uses_validated_post_resize_frames_without_live_decode(self):
        frames = np.arange(8 * 3 * 4 * 3, dtype=np.uint8).reshape(8, 3, 4, 3)
        frozen = SimpleNamespace(
            frames=frames,
            video_key="DenseVideo-LPM/videos/clip.mp4",
            total_frames=80,
            avg_fps=20.0,
            array_sha256="a" * 64,
        )
        model = self._bare_model("pyav_seek")
        model.max_image_size = 2
        model._frozen_decord_cache = mock.MagicMock()
        model._frozen_decord_cache.load.return_value = frozen

        actual = model.load_video(
            "/dataset/videos/clip.mp4",
            8,
            video_decode_backend="decord_frozen",
        )

        np.testing.assert_array_equal(actual, frames)
        model._frozen_decord_cache.load.assert_called_once_with("/dataset/videos/clip.mp4")
        self.assertEqual(model._last_video_decode_backend, "decord_frozen")
        self.assertTrue(model._last_frozen_cache_hit)
        self.assertEqual(model._last_frozen_cache_array_sha256, "a" * 64)

    def test_native_control_reports_reference_projection_ratio(self):
        model = self._bare_model("pyav_seek")
        model._last_sampled_frames = 10
        patch_embedding = torch.nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=True)
        embeddings = SimpleNamespace(patch_embedding=patch_embedding)
        vision_model = SimpleNamespace(embeddings=embeddings)
        model._model = SimpleNamespace(model=SimpleNamespace(vision_tower=SimpleNamespace(vision_model=vision_model)))
        inputs = {"pixel_values_videos": torch.zeros(1, 10, 3, 8, 8)}

        model._record_ungated_patch_metrics("video", inputs)

        self.assertEqual(model._last_recomputed_patches, 10 * 4)
        self.assertEqual(model._last_orig_patches, 10 * 4)
        self.assertEqual(model._last_reference_orig_patches, 8 * 4)
        self.assertEqual(model._last_patch_projection_recompute_ratio, 1.0)
        self.assertEqual(model._last_patch_projection_compute_ratio_vs_reference, 1.25)


if __name__ == "__main__":
    unittest.main()
