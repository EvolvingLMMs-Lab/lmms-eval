from __future__ import annotations

from pathlib import Path


def get_lmms_eval_root() -> Path:
    return Path(__file__).parent.parent


def discover_tasks(include_path: str | None = None) -> list[tuple[str, str]]:
    tasks: dict[str, str] = {}
    root = get_lmms_eval_root()
    tasks_dir = root / "tasks"
    ignore_dirs = {"__pycache__", ".ipynb_checkpoints"}

    def scan_directory(directory: Path) -> None:
        if not directory.exists():
            return

        for item in directory.iterdir():
            if item.is_dir():
                if item.name not in ignore_dirs:
                    scan_directory(item)
            elif item.suffix == ".yaml":
                try:
                    task_info = _parse_task_yaml(item)
                    if task_info:
                        task_id, display_name = task_info
                        tasks[task_id] = display_name
                except Exception:
                    pass

    scan_directory(tasks_dir)

    if include_path:
        additional_path = Path(include_path)
        if additional_path.exists():
            scan_directory(additional_path)

    return sorted(tasks.items(), key=lambda x: x[0])


def _parse_task_yaml(yaml_path: Path) -> tuple[str, str] | None:
    import yaml

    try:
        with open(yaml_path, encoding="utf-8") as f:
            content = f.read()

        if "task:" not in content and "group:" not in content:
            return None

        config = yaml.safe_load(content)
        if not isinstance(config, dict):
            return None

        if "task" in config and isinstance(config["task"], str):
            task_id = config["task"]
            display_name = _create_display_name(task_id, yaml_path)
            return (task_id, display_name)

        if "group" in config and isinstance(config["group"], str):
            group_id = config["group"]
            display_name = _create_display_name(group_id, yaml_path, is_group=True)
            return (group_id, display_name)

        return None
    except Exception:
        return None


def _create_display_name(task_id: str, yaml_path: Path, is_group: bool = False) -> str:
    parent = yaml_path.parent.name
    name = task_id.replace("_", " ").replace("-", " ").title()

    if is_group:
        return f"[Group] {name}"

    if parent.lower() != task_id.lower().replace("_", "").replace("-", ""):
        category = parent.replace("_", " ").replace("-", " ").title()
        if category.lower() not in name.lower():
            return f"{name} ({category})"

    return name


def discover_models() -> list[tuple[str, str]]:
    try:
        from lmms_eval.models import get_model_manifest, list_available_models

        models: dict[str, str] = {}
        for model_id in list_available_models(include_aliases=False):
            manifest = get_model_manifest(model_id)
            class_path = manifest.chat_class_path or manifest.simple_class_path
            class_name = class_path.rsplit(".", 1)[-1] if class_path else model_id
            models[model_id] = _create_model_display_name(model_id, class_name)

        return sorted(models.items(), key=lambda x: x[0])
    except ImportError:
        return []


def _create_model_display_name(model_id: str, class_name: str, is_chat: bool = False) -> str:
    name = class_name.replace("_", " ")
    return name


def get_popular_tasks() -> list[tuple[str, str]]:
    return [
        ("mme", "MME - Multimodal Evaluation"),
        ("mmmu_val", "MMMU Validation"),
        ("scienceqa_img", "ScienceQA (Image)"),
        ("mathvista_testmini", "MathVista TestMini"),
        ("ai2d", "AI2D"),
        ("chartqa", "ChartQA"),
        ("docvqa_val", "DocVQA Validation"),
        ("textvqa_val", "TextVQA Validation"),
        ("ocrbench", "OCRBench"),
        ("realworldqa", "RealWorldQA"),
    ]


def get_popular_models() -> list[tuple[str, str]]:
    return [
        ("openai", "OpenAI Compatible API"),
        ("qwen2_5_vl", "Qwen2.5-VL"),
        ("qwen2_5_vl_chat", "Qwen2.5-VL Chat"),
        ("llava_onevision", "LLaVA-OneVision"),
        ("llava", "LLaVA"),
        ("internvl2", "InternVL2"),
        ("claude", "Claude API"),
        ("gemini_api", "Gemini API"),
    ]


class DiscoveryCache:
    def __init__(self) -> None:
        self._tasks: list[tuple[str, str]] | None = None
        self._models: list[tuple[str, str]] | None = None
        self._all_tasks: list[tuple[str, str]] | None = None
        self._all_models: list[tuple[str, str]] | None = None

    def get_tasks(self, include_all: bool = False) -> list[tuple[str, str]]:
        if include_all:
            if self._all_tasks is None:
                self._all_tasks = discover_tasks()
            return self._all_tasks
        else:
            if self._tasks is None:
                self._tasks = get_popular_tasks()
            return self._tasks

    def get_models(self, include_all: bool = False) -> list[tuple[str, str]]:
        if include_all:
            if self._all_models is None:
                self._all_models = discover_models()
            return self._all_models
        else:
            if self._models is None:
                self._models = get_popular_models()
            return self._models

    def reload(self) -> tuple[int, int]:
        self._tasks = get_popular_tasks()
        self._models = get_popular_models()
        self._all_tasks = discover_tasks()
        self._all_models = discover_models()

        return len(self._all_tasks), len(self._all_models)

    def clear(self) -> None:
        self._tasks = None
        self._models = None
        self._all_tasks = None
        self._all_models = None


_discovery_cache = DiscoveryCache()


def get_discovery_cache() -> DiscoveryCache:
    return _discovery_cache
