from datasets import Dataset
from PIL import Image

from lmms_eval.tasks.slake import utils


def test_slake_process_docs_filters_language_and_adds_id():
    dataset = Dataset.from_list(
        [
            {"qid": 1, "q_lang": "en", "question": "Q?", "answer": "Yes"},
            {"qid": 2, "q_lang": "zh", "question": "问?", "answer": "是"},
        ]
    )

    processed = utils.slake_process_docs_en(dataset)

    assert len(processed) == 1
    assert processed[0]["id"] == "1"
    assert processed[0]["q_lang"] == "en"


def test_slake_doc_to_text_uses_prompts():
    doc = {"question": "What modality is used?"}

    prompt = utils.slake_doc_to_text(doc, {"pre_prompt": "Question: ", "post_prompt": "\nAnswer:"})

    assert prompt == "Question: What modality is used?\nAnswer:"


def test_slake_doc_to_visual_accepts_pil_image():
    image = Image.new("RGB", (4, 4), color="white")
    doc = {"image": image, "img_name": "unused.jpg"}

    visuals = utils.slake_doc_to_visual(doc)

    assert len(visuals) == 1
    assert visuals[0].mode == "RGB"


def test_slake_doc_to_messages_contains_image_then_text():
    image = Image.new("RGB", (4, 4), color="white")
    doc = {"image": image, "img_name": "unused.jpg", "question": "What modality is used?"}

    messages = utils.slake_doc_to_messages(doc, {"post_prompt": "\nAnswer:"})

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][0]["url"].mode == "RGB"
    assert messages[0]["content"][1] == {"type": "text", "text": "What modality is used?\nAnswer:"}


def test_slake_resolve_zip_member_accepts_imgs_prefix(monkeypatch, tmp_path):
    import zipfile

    archive_path = tmp_path / "imgs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("imgs/xmlab102/source.jpg", b"demo")
    monkeypatch.setattr(utils, "_slake_image_zip_path", lambda: str(archive_path))

    assert utils._resolve_zip_member("xmlab102/source.jpg") == "imgs/xmlab102/source.jpg"


def test_slake_normalize_answer_handles_case_punctuation_and_yes_no():
    assert utils.normalize_answer(" Yes. ") == "yes"
    assert utils.normalize_answer("是的。") == "yes"
    assert utils.normalize_answer("不包含") == "no"
    assert utils.normalize_answer("不可以") == "no"
    assert utils.normalize_answer("不正常") == "no"
    assert utils.normalize_answer("Left Lung, Right") == "left lung right"
    assert utils.normalize_answer("CT (computed tomography)") == "ct"
    assert utils.normalize_answer("X-Ray") == utils.normalize_answer("xray")
    assert utils.normalize_answer("胸腔（肺部）") == "胸腔"


def test_slake_process_results_exact_match_with_metadata():
    doc = {
        "qid": 7,
        "q_lang": "en",
        "answer": "CT",
        "answer_type": "OPEN",
        "modality": "CT",
        "content_type": "Modality",
    }

    result = utils.slake_process_results(doc, ["ct"])
    record = result["slake_accuracy"]

    assert record["is_correct"] is True
    assert record["language"] == "en"
    assert record["answer_type"] == "open"
    assert record["modality"] == "ct"
    assert record["content_type"] == "modality"


def test_slake_aggregate_accuracy_and_subgroups():
    rows = [
        {"is_correct": True, "language": "en", "answer_type": "open", "modality": "ct", "content_type": "modality"},
        {"is_correct": False, "language": "zh", "answer_type": "closed", "modality": "mri", "content_type": "organ"},
        {"is_correct": True, "language": "zh", "answer_type": "closed", "modality": "mri", "content_type": "organ"},
    ]

    assert utils.slake_aggregate_accuracy(rows) == 2 / 3
    assert utils.slake_aggregate_open_accuracy(rows) == 1.0
    assert utils.slake_aggregate_closed_accuracy(rows) == 0.5
    assert utils.slake_aggregate_ct_accuracy(rows) == 1.0
    assert utils.slake_aggregate_mri_accuracy(rows) == 0.5
    assert utils.slake_aggregate_organ_accuracy(rows) == 0.5
