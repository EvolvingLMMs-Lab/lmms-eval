import pytest

from lmms_eval.tasks.imavb import utils


def make_doc(correct_answer="B"):
    return {
        "video": "videos/clip.mp4",
        "question": "When the man in the dark suit faces the younger man, what colour is the dress of the woman in the doorway?",
        "option_a": "Black",
        "option_b": "Red",
        "option_c": "White",
        "option_d": "Dark red",
        "option_e": "The visual detail in the question is incorrect",
        "option_f": "The audio detail in the question is incorrect",
        "correct_answer": correct_answer,
    }


def test_doc_to_text_lists_six_options_in_fixed_order():
    text = utils.imavb_doc_to_text(make_doc(), {"pre_prompt": "", "post_prompt": "\n\nAnswer with the option's letter from the given choices directly."})
    assert text.startswith("Question:\nWhen the man in the dark suit")
    assert "\n\nOptions:\nA. Black\nB. Red\nC. White\nD. Dark red\nE. The visual detail in the question is incorrect\nF. The audio detail in the question is incorrect" in text
    assert text.endswith("\n\nAnswer with the option's letter from the given choices directly.")


def test_doc_to_visual_resolves_clip_under_cache_dir(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "clip.mp4").write_bytes(b"")
    monkeypatch.setattr(utils, "CACHE_DIR", str(tmp_path))
    assert utils.imavb_doc_to_visual(make_doc()) == [str(tmp_path / "videos" / "clip.mp4")]


def test_doc_to_visual_raises_for_missing_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "CACHE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        utils.imavb_doc_to_visual(make_doc())


@pytest.mark.parametrize(
    "response, expected",
    [
        ("E", "E"),
        ("F.", "F"),
        ("(B)", "B"),
        ("Answer: C", "C"),
        ("E. The visual detail in the question is incorrect", "E"),
        ("The audio detail in the question is incorrect.", "F"),
        ("The dress is white.", "C"),
        ("The dress is red or dark red.", ""),
        ("A or E", ""),
        ("G", ""),
        ("I am not sure.", ""),
        ("", ""),
    ],
)
def test_extract_answer(response, expected):
    assert utils.imavb_extract_answer(response, make_doc()) == expected


def test_process_results_scores_accuracy():
    assert utils.imavb_process_results(make_doc("E"), ["E"]) == {"accuracy": 1.0}
    assert utils.imavb_process_results(make_doc("E"), ["B"]) == {"accuracy": 0.0}
    assert utils.imavb_process_results(make_doc("F"), [""]) == {"accuracy": 0.0}
