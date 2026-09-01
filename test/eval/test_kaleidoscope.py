import pytest
from PIL import Image

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.kaleidoscope import utils


def _doc(language="en", answer=1, image="data/exam/images/q1.png", options=None):
    return {
        "language": language,
        "country": "India",
        "file_name": "xl_2022.pdf",
        "source": "GATE",
        "license": "cc-by",
        "level": "university",
        "category_en": "Biology",
        "category_original_lang": "Biology",
        "general_category_en": "STEM",
        "original_question_num": "11",
        "question": "Which structure is labelled X?",
        "options": options if options is not None else ["Nucleus", "Ribosome", "Golgi", "Vacuole"],
        "answer": answer,
        "image_png": "q1.png",
        "image_information": "essential",
        "image_type": "diagram",
        "parallel_question_id": "",
        "image": image,
    }


def _write_png(root, relative_path, size=(8, 6), color=(10, 20, 30)):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        "kaleidoscope_direct",
        "kaleidoscope_multimodal",
        "kaleidoscope_text_only",
        "kaleidoscope_cot",
        "kaleidoscope_multimodal_cot",
    ],
)
def test_tasks_are_registered(task):
    assert task in TaskManager("WARNING").all_subtasks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_direct_prompt_uses_in_language_keywords_and_json_system_message():
    prompt = utils.kaleidoscope_doc_to_text(_doc(language="bn"), {"prompt_type": "direct"})

    assert '{"choice": "The correct option (e.g., A, B, C, or D)"}' in prompt
    assert "\\ONLY" not in prompt  # reference escape-sequence typo must not leak through
    assert "প্রশ্ন: Which structure is labelled X?" in prompt
    assert "বিকল্প:" in prompt
    assert "A. Nucleus" in prompt
    assert "D. Vacuole" in prompt
    assert prompt.endswith("\nANSWER:")


def test_cot_prompt_uses_in_language_system_message_and_no_answer_cue():
    prompt = utils.kaleidoscope_doc_to_text(_doc(language="es"), {"prompt_type": "cot"})

    assert prompt.startswith(utils.INSTRUCTIONS_COT["es"])
    assert "<ANSWER> X </ANSWER>" in prompt
    assert "Options:" in prompt
    assert not prompt.endswith("ANSWER:")


def test_cot_prompt_rejects_unknown_language():
    with pytest.raises(ValueError, match="chain-of-thought instruction"):
        utils.kaleidoscope_doc_to_text(_doc(language="zz"), {"prompt_type": "cot"})


def test_every_language_has_keywords_and_cot_instruction():
    assert set(utils.KEYWORDS) == set(utils.LANGUAGES)
    assert set(utils.INSTRUCTIONS_COT) == set(utils.LANGUAGES)


def test_doc_to_messages_puts_question_image_first_for_direct(monkeypatch, tmp_path):
    monkeypatch.setenv("KALEIDOSCOPE_DATA_ROOT", str(tmp_path))
    _write_png(tmp_path, "data/exam/images/q1.png")

    messages = utils.kaleidoscope_doc_to_messages(_doc(), {"prompt_type": "direct"})

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][-1]["text"] == "\nANSWER:"


def test_doc_to_messages_puts_question_text_first_for_cot(monkeypatch, tmp_path):
    monkeypatch.setenv("KALEIDOSCOPE_DATA_ROOT", str(tmp_path))
    _write_png(tmp_path, "data/exam/images/q1.png")

    messages = utils.kaleidoscope_doc_to_messages(_doc(), {"prompt_type": "cot"})

    assert messages[1]["content"][0] == {"type": "text", "text": "Which structure is labelled X?"}
    assert messages[1]["content"][1]["type"] == "image"


def test_doc_to_messages_interleaves_image_options(monkeypatch, tmp_path):
    monkeypatch.setenv("KALEIDOSCOPE_DATA_ROOT", str(tmp_path))
    _write_png(tmp_path, "data/exam/images/q1.png")
    _write_png(tmp_path, "data/exam/images/opt_a.png")

    doc = _doc(options=["data/exam/images/opt_a.png", "Ribosome"])
    content = utils.kaleidoscope_doc_to_messages(doc, {"prompt_type": "direct"})[1]["content"]
    images = [item for item in content if item["type"] == "image"]

    assert len(images) == 2  # question image + one option image
    assert {"type": "text", "text": "A."} in content
    assert {"type": "text", "text": "B. Ribosome"} in content


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------


def test_doc_to_visual_resizes_to_configured_edge_length(monkeypatch, tmp_path):
    monkeypatch.setenv("KALEIDOSCOPE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("KALEIDOSCOPE_IMAGE_SIZE", "512")
    _write_png(tmp_path, "data/exam/images/q1.png", size=(40, 10))

    visuals = utils.kaleidoscope_doc_to_visual(_doc())

    assert len(visuals) == 1
    assert visuals[0].size == (512, 512)
    assert visuals[0].mode == "RGB"


def test_doc_to_visual_can_keep_native_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("KALEIDOSCOPE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("KALEIDOSCOPE_IMAGE_SIZE", "0")
    _write_png(tmp_path, "data/exam/images/q1.png", size=(40, 10))

    assert utils.kaleidoscope_doc_to_visual(_doc())[0].size == (40, 10)


def test_doc_to_visual_is_empty_for_text_only_questions(monkeypatch, tmp_path):
    monkeypatch.setenv("KALEIDOSCOPE_DATA_ROOT", str(tmp_path))

    assert utils.kaleidoscope_doc_to_visual(_doc(image=None)) == []


def test_doc_to_visual_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setenv("KALEIDOSCOPE_DATA_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="must be relative"):
        utils.kaleidoscope_doc_to_visual(_doc(image="../../secret.png"))


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ('{"choice": "B"}', 1),
        ('```json\n{"choice": "C"}\n```', 2),
        ('Sure!\n{"choice": "D"}', 3),
        ('{"choice": "B. Ribosome"}', 1),
        ('{"choice": "b"}', 1),
        ("B", 1),
        ("2", 1),
    ],
)
def test_direct_extraction(response, expected):
    assert utils.extract_choice(response, 4, "direct") == expected


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Step one... <ANSWER> C </ANSWER>", 2),
        ("<ANSWER>A</ANSWER>", 0),
        ("<ANSWER> b </ANSWER>", 1),
        ("first <ANSWER> A </ANSWER> then <ANSWER> D </ANSWER>", 3),
    ],
)
def test_cot_extraction(response, expected):
    assert utils.extract_choice(response, 4, "cot") == expected


def test_non_latin_choice_letters_are_mapped():
    assert utils.extract_choice('{"choice": "Б"}', 4, "direct") == 1
    assert utils.extract_choice("<ANSWER> د </ANSWER>", 4, "cot") == 3


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        "I cannot answer this question.",
        "<ANSWER></ANSWER>",
        '{"choice": "A or B"}',
        '{"answer": "B"}',
    ],
)
def test_unparseable_responses_return_none(response):
    assert utils.extract_choice(response, 4, "direct") is None


def test_out_of_range_letters_are_rejected():
    assert utils.extract_choice('{"choice": "D"}', 2, "direct") is None


def test_lenient_extraction_is_opt_in():
    response = "The correct answer is (C) because the diagram shows a Golgi body."

    assert utils.extract_choice(response, 4, "direct") is None
    assert utils.extract_choice(response, 4, "direct", lenient=True) == 2


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_process_results_emits_all_three_metrics():
    scored = utils.kaleidoscope_process_results(_doc(answer=1), ['{"choice": "B"}'])

    assert set(scored) == {"kaleidoscope_acc", "kaleidoscope_valid_acc", "kaleidoscope_format_error"}
    record = scored["kaleidoscope_acc"]
    assert record["language"] == "English"
    assert record["general_category"] == "STEM"
    assert record["is_multimodal"] is True
    assert record["prediction"] == 1
    assert record["valid"] is True
    assert record["correct"] is True


def test_format_error_counts_as_wrong_but_not_as_valid():
    record = utils.kaleidoscope_process_results(_doc(answer=1), ["no idea"])["kaleidoscope_acc"]

    assert record["prediction"] is None
    assert record["valid"] is False
    assert record["correct"] is False


def test_cot_process_results_reads_answer_tags():
    record = utils.kaleidoscope_process_results_cot(_doc(answer=2), ["reasoning <ANSWER> C </ANSWER>"])["kaleidoscope_acc"]

    assert record["prediction"] == 2
    assert record["correct"] is True


def _records(*specs):
    """Build metric records from (language, gold, response) triples."""
    return [utils.kaleidoscope_process_results(_doc(language=language, answer=answer), [response])["kaleidoscope_acc"] for language, answer, response in specs]


def test_accuracy_is_macro_averaged_over_languages():
    # English: 2/2 correct.  Bengali: 0/4 correct.  Micro would be 33.3%.
    records = _records(
        ("en", 0, '{"choice": "A"}'),
        ("en", 1, '{"choice": "B"}'),
        ("bn", 0, '{"choice": "B"}'),
        ("bn", 0, '{"choice": "B"}'),
        ("bn", 0, '{"choice": "B"}'),
        ("bn", 0, '{"choice": "B"}'),
    )

    assert utils.kaleidoscope_aggregate_accuracy(records) == pytest.approx(50.0)


def test_valid_accuracy_excludes_format_errors():
    records = _records(
        ("en", 0, '{"choice": "A"}'),
        ("en", 0, "I refuse to answer"),
    )

    assert utils.kaleidoscope_aggregate_accuracy(records) == pytest.approx(50.0)
    assert utils.kaleidoscope_aggregate_valid_accuracy(records) == pytest.approx(100.0)
    assert utils.kaleidoscope_aggregate_format_error(records) == pytest.approx(50.0)


def test_format_error_is_micro_averaged_over_all_samples():
    records = _records(
        ("en", 0, "no"),
        ("bn", 0, '{"choice": "A"}'),
        ("bn", 0, '{"choice": "A"}'),
        ("bn", 0, '{"choice": "A"}'),
    )

    assert utils.kaleidoscope_aggregate_format_error(records) == pytest.approx(25.0)


def test_aggregations_handle_empty_results():
    assert utils.kaleidoscope_aggregate_accuracy([]) == 0.0
    assert utils.kaleidoscope_aggregate_valid_accuracy([]) == 0.0
    assert utils.kaleidoscope_aggregate_format_error([]) == 0.0


# ---------------------------------------------------------------------------
# Split filters
# ---------------------------------------------------------------------------


def test_split_filters_partition_the_dataset():
    import datasets

    dataset = datasets.Dataset.from_list([_doc(), _doc(image=None), _doc(image="")])

    assert len(utils.kaleidoscope_filter_multimodal(dataset)) == 1
    assert len(utils.kaleidoscope_filter_text_only(dataset)) == 2
