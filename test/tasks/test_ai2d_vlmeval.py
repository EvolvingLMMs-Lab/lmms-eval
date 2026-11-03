import base64
import csv
import io
from pathlib import Path

import pytest
from datasets import Dataset, load_dataset
from PIL import Image

from lmms_eval.tasks.ai2d import utils
from lmms_eval.utils import load_yaml_config


def encoded_image(color):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def row(index=1, image=None, answer="D", **kwargs):
    return {
        "index": index,
        "question": "Which stage?",
        "A": "fruit",
        "B": "growth",
        "C": "flowering",
        "D": "transplant",
        "answer": answer,
        "image": image or encoded_image("red"),
        "abcLabel": False,
        "image_path": "ai2d/images/1287.png",
        **kwargs,
    }


def test_question_specific_images_and_legacy_contract():
    source = Dataset.from_list([row(), row(2, encoded_image("blue"), "A", abcLabel=True, image_path="ai2d/images_abc/1287.png")])
    result = utils.ai2d_vlmeval_process_docs(source)
    assert utils.ai2d_vlmeval_doc_to_visual(result[0])[0].getpixel((0, 0)) == (255, 0, 0)
    assert utils.ai2d_vlmeval_doc_to_visual(result[1])[0].getpixel((0, 0)) == (0, 0, 255)
    legacy = {"question": "Which stage?", "options": ["fruit", "growth", "flowering", "transplant"], "answer": "3"}
    config = {"prompt_format": "mcq", "pre_prompt": "", "post_prompt": ""}
    assert utils.ai2d_doc_to_text(result[0], config) == utils.ai2d_doc_to_text(legacy, config)
    assert utils.ai2d_doc_to_target(result[0], "mcq") == "D"
    assert result[0]["index"] == 1 and result[1]["abcLabel"] is True


def test_references_resolved_before_selection():
    result = utils.ai2d_vlmeval_process_docs(Dataset.from_list([row(1, "2"), row(2, "3"), row(3, encoded_image("blue"))]))
    selected = result.select([0])
    assert utils.ai2d_vlmeval_doc_to_visual(selected[0])[0].getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize("rows", [[row(1), row(1)], [row(1, "99")], [row(1, "2"), row(2, "1")]])
def test_invalid_image_reference_fails(rows):
    with pytest.raises(ValueError):
        utils.ai2d_vlmeval_process_docs(Dataset.from_list(rows))


@pytest.mark.parametrize("answer", ["E", "", None])
def test_invalid_answer_fails(answer):
    with pytest.raises(ValueError):
        utils.ai2d_vlmeval_process_docs(Dataset.from_list([row(answer=answer)]))


def test_invalid_base64_fails():
    with pytest.raises(ValueError):
        utils.ai2d_vlmeval_doc_to_visual({"image": "!" * 100})


def test_task_config_reuses_legacy_scoring_without_changing_legacy():
    directory = Path(utils.__file__).parent
    legacy = load_yaml_config(str(directory / "ai2d.yaml"), mode="simple")
    variant = load_yaml_config(str(directory / "ai2d_vlmeval.yaml"), mode="simple")
    assert variant["task"] == "ai2d_vlmeval"
    assert legacy["dataset_path"] == "lmms-lab-encoder/ai2d"
    assert variant["dataset_path"] == "csv"
    assert "/mm-eval/VLMEvalKit/resolve/772639df87331ba19da76ea9cc5fa77836a15a18/AI2D_TEST.tsv" in variant["dataset_kwargs"]["data_files"]["test"]
    assert variant["dataset_kwargs"]["delimiter"] == "\t"
    assert variant["dataset_kwargs"]["keep_default_na"] is False
    assert variant["tag"] == []
    for key in ("generation_kwargs", "metric_list", "lmms_eval_specific_kwargs", "model_specific_target_kwargs"):
        assert variant[key] == legacy[key]
    # YAML constructors return distinct ScalarNode objects on each load.
    assert repr(variant["filter_list"]) == repr(legacy["filter_list"])


@pytest.mark.parametrize("missing", ["A", "B", "C", "D"])
def test_missing_option_fails(missing):
    sample = row()
    sample[missing] = None
    with pytest.raises(ValueError):
        utils.ai2d_vlmeval_process_docs(Dataset.from_list([sample]))


def test_literal_na_option_survives_csv_loading(tmp_path):
    path = tmp_path / "ai2d.tsv"
    sample = row()
    sample["D"] = "NA"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample, delimiter="\t")
        writer.writeheader()
        writer.writerow(sample)
    loaded = load_dataset("csv", data_files=str(path), delimiter="\t", keep_default_na=False, split="train")
    assert utils.ai2d_vlmeval_process_docs(loaded)[0]["options"][3] == "NA"
