import unittest
from pathlib import Path

import yaml

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.av_asr import utils as av_asr_utils
from lmms_eval.tasks.countix import utils as countix_utils
from lmms_eval.tasks.ovr_kinetics import utils as ovr_kinetics_utils
from lmms_eval.tasks.repcount import utils as repcount_utils
from lmms_eval.tasks.ssv2 import utils as ssv2_utils
from lmms_eval.tasks.vggsound import utils as vggsound_utils


class TestNewBenchmarkTaskRegistration(unittest.TestCase):
    def test_new_tasks_are_registered(self):
        task_manager = TaskManager()
        expected_tasks = {"repcount", "countix", "ovr_kinetics", "ssv2", "vggsound", "av_asr", "neptune"}
        missing_tasks = expected_tasks.difference(task_manager.all_tasks)
        self.assertFalse(missing_tasks, f"Missing new benchmark tasks: {sorted(missing_tasks)}")


class TestRepCountUtils(unittest.TestCase):
    def test_repcount_process_results(self):
        doc = {"count": 5, "video": "demo.mp4", "question": "How many reps?"}
        result = repcount_utils.repcount_process_results(doc, ["6"])
        self.assertAlmostEqual(result["mae_norm"], 1 / 5.1)
        self.assertEqual(result["obo"], 1.0)


class TestCountixUtils(unittest.TestCase):
    def test_countix_process_results(self):
        doc = {"count": 3, "video": "demo.mp4"}
        result = countix_utils.countix_process_results(doc, ["1"])
        self.assertAlmostEqual(result["mae_norm"], 2 / 3.1)
        self.assertEqual(result["obo"], 0.0)


class TestOVRKineticsUtils(unittest.TestCase):
    def test_ovr_kinetics_process_results(self):
        doc = {"count": 4, "video": "demo.mp4", "text_description": "jumping"}
        result = ovr_kinetics_utils.ovr_kinetics_process_results(doc, ["5"])
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["obo"], 1.0)


class TestSSV2Utils(unittest.TestCase):
    def test_ssv2_process_results(self):
        doc = {"text": "moving drawer of night stand", "video": "41775.webm"}
        result = ssv2_utils.ssv2_process_results(doc, ["moving drawer of night stand"])
        self.assertEqual(result["acc"], 1.0)


class TestVGGSoundUtils(unittest.TestCase):
    def test_vggsound_process_results(self):
        doc = {"label": "playing acoustic guitar", "audio": "demo.wav"}
        result = vggsound_utils.vggsound_process_results(doc, ["playing acoustic guitar"])
        self.assertEqual(result["acc"], 1.0)


class TestAVASRUtils(unittest.TestCase):
    def test_av_asr_process_and_aggregation(self):
        doc = {"transcript": "hello world", "video": "demo.mp4"}
        processed = av_asr_utils.av_asr_process_results(doc, ["hello world"])
        self.assertIn("wer", processed)
        score = av_asr_utils.av_asr_wer([processed["wer"]])
        self.assertEqual(score, 0.0)


class TestNewBenchmarkTaskConfigSources(unittest.TestCase):
    class _FunctionTag:
        def __init__(self, value):
            self.value = value

    @staticmethod
    def _function_constructor(loader, node):
        return TestNewBenchmarkTaskConfigSources._FunctionTag(loader.construct_scalar(node))

    @staticmethod
    def _load_yaml(file_path):
        loader = yaml.SafeLoader
        loader_copy = type("SafeLoaderCopy", (loader,), {})
        loader_copy.add_constructor("!function", TestNewBenchmarkTaskConfigSources._function_constructor)
        with file_path.open("r", encoding="utf-8") as handle:
            return yaml.load(handle, Loader=loader_copy)

    def test_fullset_dataset_paths(self):
        root = Path(__file__).resolve().parents[2]
        expected = {
            "lmms_eval/tasks/egotempo/egotempo.yaml": "lmms-lab-eval/egotempo",
            "lmms_eval/tasks/repcount/repcount.yaml": "lmms-lab-eval/repcount",
            "lmms_eval/tasks/countix/countix.yaml": "lmms-lab-eval/countix",
            "lmms_eval/tasks/ovr_kinetics/ovr_kinetics.yaml": "lmms-lab-eval/ovr_kinetics",
            "lmms_eval/tasks/ssv2/ssv2.yaml": "lmms-lab-eval/ssv2",
            "lmms_eval/tasks/vggsound/vggsound.yaml": "lmms-lab-eval/vggsound",
            "lmms_eval/tasks/av_asr/av_asr.yaml": "lmms-lab-eval/av_asr",
        }

        for rel_path, dataset_path in expected.items():
            path = root / rel_path
            with self.subTest(task_yaml=str(path)):
                content = self._load_yaml(path)
                self.assertEqual(content.get("dataset_path"), dataset_path)
                data_files = content.get("dataset_kwargs", {}).get("data_files")
                self.assertFalse(data_files, f"{rel_path} still uses local data_files")


if __name__ == "__main__":
    unittest.main()
