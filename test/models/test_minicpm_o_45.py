import inspect
import unittest

from lmms_eval.models import get_model, get_model_manifest
from lmms_eval.models.chat.minicpm_o_45 import MiniCPM_O_45Chat
from lmms_eval.models.simple.minicpm_o_45 import MiniCPM_O_45
from lmms_eval.models.simple.minicpm_o_45_duplex import MiniCPM_O_45Duplex


class TestMiniCPMO45Registration(unittest.TestCase):
    def test_registry_resolves_simple_and_chat_adapters(self):
        manifest = get_model_manifest("minicpm_o_45")

        self.assertEqual(manifest.simple_class_path, "lmms_eval.models.simple.minicpm_o_45.MiniCPM_O_45")
        self.assertEqual(manifest.chat_class_path, "lmms_eval.models.chat.minicpm_o_45.MiniCPM_O_45Chat")
        # the registry prefers the chat class unless force_simple is set
        self.assertIs(get_model("minicpm_o_45"), MiniCPM_O_45Chat)
        self.assertIs(get_model("minicpm_o_45", force_simple=True), MiniCPM_O_45)

    def test_registry_resolves_duplex_adapter(self):
        manifest = get_model_manifest("minicpm_o_45_duplex")

        self.assertEqual(manifest.simple_class_path, "lmms_eval.models.simple.minicpm_o_45_duplex.MiniCPM_O_45Duplex")
        self.assertIsNone(manifest.chat_class_path)

    def test_defaults_target_minicpm_o_4_5_checkpoint(self):
        params = inspect.signature(MiniCPM_O_45.__init__).parameters

        self.assertEqual(params["pretrained"].default, "openbmb/MiniCPM-o-4_5")
        self.assertTrue(params["init_vision"].default)
        self.assertTrue(params["init_audio"].default)
        self.assertFalse(params["init_tts"].default)
        self.assertFalse(params["enable_thinking"].default)


class TestMiniCPMO45Helpers(unittest.TestCase):
    def test_truncate_until_stops_at_earliest_marker(self):
        self.assertEqual(MiniCPM_O_45._truncate_until("A. extra", ["."]), "A")
        self.assertEqual(MiniCPM_O_45._truncate_until("no markers", ["."]), "no markers")

    def test_split_audio_chunks_pads_final_chunk(self):
        import numpy as np

        chunks = MiniCPM_O_45Duplex._split_audio_chunks(np.zeros(16000 * 2 + 5, dtype=np.float32))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) == 16000 for chunk in chunks))

    def test_duplex_classification(self):
        import numpy as np

        streams = MiniCPM_O_45Duplex._classify_media(
            ["video.mp4", {"array": np.zeros(1600, dtype="float32"), "sampling_rate": 16000}]
        )
        self.assertEqual([s["kind"] for s in streams], ["video", "audio"])


if __name__ == "__main__":
    unittest.main()
