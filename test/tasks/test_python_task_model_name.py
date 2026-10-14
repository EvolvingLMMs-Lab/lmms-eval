from lmms_eval.tasks import TaskManager


def test_python_task_receives_model_name():
    class ModelAwareTask:
        def __init__(self, config=None, model_name=None):
            self.model_name = model_name

    manager = TaskManager(model_name="openai")
    tasks = manager.load_task_or_group([{"task": "model_aware_test", "class": ModelAwareTask}])
    assert tasks["model_aware_test"].model_name == "openai"


def test_legacy_python_task_constructor_stays_compatible():
    class LegacyTask:
        def __init__(self, config=None):
            self.marker = config["marker"]

    manager = TaskManager(model_name="openai")
    tasks = manager.load_task_or_group([{"task": "legacy_test", "class": LegacyTask, "marker": "kept"}])
    assert tasks["legacy_test"].marker == "kept"
