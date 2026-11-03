from pathlib import Path

import yaml


def _load_template(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        content = "".join(line for line in handle if "!function" not in line)
    return yaml.safe_load(content)


def test_refspatial_preserves_image_columns_for_result_processing():
    root = Path(__file__).resolve().parents[2]
    config = _load_template(root / "lmms_eval/tasks/refspatial/_default_template_yaml")

    assert config["full_docs"] is True
    assert config["process_results_use_image"] is True


def test_where2place_preserves_image_columns_for_result_processing():
    root = Path(__file__).resolve().parents[2]
    config = _load_template(root / "lmms_eval/tasks/where2place/_default_template_yaml")

    assert config["full_docs"] is True
    assert config["process_results_use_image"] is True
