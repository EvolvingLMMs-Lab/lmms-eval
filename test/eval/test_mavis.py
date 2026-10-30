import pytest
from PIL import Image

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.mavis import utils


def _doc():
    return {
        "Q": "Should I trim this tree myself?",
        "image": Image.new("RGB", (512, 384), "green"),
        "verifiable_facts": ["Large limbs need a professional.", "Small branches can be trimmed with a pole saw."],
        "documents_id": ["1420-0-0", "38250", "1420-1-0"],
        "documents_type": ["text", "image", "text"],
        "documents_text": ["Hire an arborist for limbs over 4 inches.", "", "A pole saw handles small branches."],
        "document_images": [Image.new("RGB", (512, 512), "brown")],
    }


def _scripted_judge(contents):
    """Supports sentences that mention 'arborist', calls every sentence relevant and every fact partially addressed."""
    responses = []
    for content in contents:
        text = "".join(part.get("text", "") for part in content)
        if text.startswith("Instruction:\n1. You will be given a question, a statement"):
            responses.append("Answer: fully support" if "arborist" in text.split("Documents:")[0] else "Answer: not support")
        elif text.startswith("Instructions:"):
            responses.append("Label: Fully relevant")
        else:
            responses.append("Label: Partially addressed")
    return responses


def test_prompt_interleaves_numbered_documents():
    doc = _doc()
    content = utils.mavis_doc_to_messages(doc)[0]["content"]

    assert [part["type"] for part in content] == ["text", "image", "text", "text", "text", "image", "text", "text"]
    assert content[0]["text"].startswith("Based on the documents") and content[0]["text"].endswith("Question: ")
    assert content[2]["text"] == "Should I trim this tree myself?\nDocument:"
    assert content[3]["text"] == "\n[1]: Hire an arborist for limbs over 4 inches."
    assert content[4]["text"] == "\n[2]: " and content[5]["url"].size == (512, 512)
    assert content[-1]["text"] == "\n\nAnswer:"
    assert len(utils.mavis_doc_to_visual(doc)) == 2
    assert "\n[2]: <image 2>" in utils.mavis_doc_to_text(doc)


def test_text_prompt_does_not_need_the_image_columns():
    doc = _doc()
    no_media = {k: v for k, v in doc.items() if k not in ("image", "document_images")}

    assert utils.mavis_doc_to_text(no_media) == utils.mavis_doc_to_text(doc)


def test_scoring_without_images_fails_loudly():
    no_media = {k: v for k, v in _doc().items() if k not in ("image", "document_images")}

    with pytest.raises(ValueError, match="process_results_use_image"):
        utils.mavis_process_results(no_media, ["An answer [1]."])


def test_citations_are_parsed_and_moved_back_to_their_sentence():
    pairs = utils.answer_sentences("Hire an arborist [1][2]. Trim small branches. [3] Done [9].", ["1420-0-0", "38250", "1420-1-0"])

    assert pairs == [("Hire an arborist.", ["1420-0-0", "38250"]), ("Trim small branches.", ["1420-1-0"]), ("Done.", [])]


def test_scores_follow_the_reference_definitions(monkeypatch):
    monkeypatch.setattr(utils, "_ask_all", _scripted_judge)
    results = utils.mavis_process_results(_doc(), ["Hire an arborist for big limbs [1][2]. Use a pole saw [3]. Be careful."])

    # Recall: 1 (supported) + 0 + 0 (uncited) over 3 sentences. Precision: sentence 1 cites two docs, each judged
    # with the sentence alone (both supported); sentence 2 is unsupported -> mean(1, 0).
    assert results["mavis_recall"] == pytest.approx(1 / 3)
    assert results["mavis_precision"] == pytest.approx(0.5)
    assert results["mavis_groundedness"] == pytest.approx(0.4)
    assert results["mavis_completeness"] == pytest.approx(0.5)
    assert results["mavis_relevance"] == pytest.approx(1.0)
    assert results["mavis_informativeness"] == pytest.approx(2 / 3)


def test_empty_answer_scores_zero_without_judge_calls(monkeypatch):
    monkeypatch.setattr(utils, "_ask_all", lambda contents: pytest.fail("judge should not be called"))
    results = utils.mavis_process_results(_doc(), [""])

    assert set(results.values()) == {0.0}
    assert utils.mavis_aggregate([0.5, None, 1.0]) == pytest.approx(75.0)


def test_judge_images_use_low_detail_like_the_reference():
    part = utils._image_part(Image.new("RGB", (1024, 768)))

    assert part["image_url"]["detail"] == "low"
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_label_scoring():
    assert utils._label_score("The claim is backed. Answer: Fully support", "answer:") == 1.0
    assert utils._label_score("Label: Partially relevant", "label:") == 0.5
    assert utils._label_score("Label: Not addressed", "label:") == 0.0
    assert utils._label_score("no label at all", "label:") == 0.0


def test_mini_subset_is_fixed_and_keeps_the_domain_mix():
    domains = ["a"] * 500 + ["b"] * 300 + ["c"] * 199 + ["d"]
    qids = [str(i // 2) for i in range(1000)]  # some questions have two answers
    indices = utils.mini_indices(domains, qids)

    assert len(indices) == 200 and len(set(indices)) == 200
    assert indices == utils.mini_indices(domains, qids)
    counts = {d: sum(domains[i] == d for i in indices) for d in "abcd"}
    assert counts == {"a": 100, "b": 60, "c": 40, "d": 0}


def test_tasks_are_discoverable_and_keep_images_for_scoring():
    manager = TaskManager()
    configs = {"gt": "gt_rag", "multirag": "multi_rag", "textrag": "text_rag", "imagerag": "image_rag"}
    for setting, dataset_name in configs.items():
        for task in (f"mavis_{setting}", f"mavis_{setting}_mini"):
            assert task in manager.all_tasks
            config = manager._get_config(task)
            assert config["dataset_name"] == dataset_name
            assert config["process_results_use_image"] is True
            assert ("process_docs" in config) == task.endswith("_mini")
