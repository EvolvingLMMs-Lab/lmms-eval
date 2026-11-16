"""Native option-letter parsers that read the final answer first and keep their own heuristic as the fallback.

Three properties are checked for every parser:

* a declared final answer in a common format is read;
* a response the final-answer step does not read is parsed exactly as before;
* a bare option letter gives the same result as before.
"""

import importlib
import random
import sys
from unittest import mock

import pytest

from lmms_eval.tasks._task_utils.mcq_extract import extract_final_mcq_answer

INDEX2ANS = {"A": "a red car", "B": "a blue bike", "C": "a green tree", "D": "an old book", "E": "a tall tower", "F": "a small boat", "G": "a white cup", "H": "a black hat"}


def _scored(score, letters):
    """The offered letters that a scorer accepts as the gold answer of a response."""
    return "".join(letter for letter in letters if score(letter))


def _texts(letters):
    return [INDEX2ANS[letter] for letter in letters]


def _tempcompass(parser, response, letters):
    def rating(gold):
        doc = {"answer": f"{gold}. {INDEX2ANS[gold]}", "video_id": "v", "question": "q", "dim": "speed"}
        return parser(doc, [response])["avg_accuracy"]["rating"]

    with mock.patch.object(sys.modules[parser.__module__], "get_eval_result", return_value=("Incorrect", 0)):
        return _scored(lambda gold: rating(gold) == 1, letters)


def _ai2d(parser, response, letters):
    regex_filter = parser(regex_pattern=r"([A-Z])\.", group_select=0, ignore_case=True, ignore_punctuation=True)
    return regex_filter.apply([[response]], [{"options": _texts(letters)}])[0]


def _seed(parser, response, letters):
    return parser({"answer": "A", "data_type": "x", "question_id": 0}, [response])["seed_x"]["pred"]


# (module under lmms_eval.tasks, function, call style, offered letters)
#   "choices+index2ans": f(response, all_choices, index2ans)
#   "choices":           f(response, all_choices)
#   "response":          f(response)
#   "letters":           f(response, "ABCD")
#   "option_texts":      f(response, [option texts])
#   a callable:          style(f, response, letters); scorers return the gold letters they accept
PARSERS = [
    ("_task_utils.mmmu_mcq_utils", "parse_mmmu_multi_choice_response", "choices+index2ans", "ABCD"),
    ("_task_utils.mmmu_mcq_utils", "parse_jmmmu_multi_choice_response", "choices+index2ans", "ABCD"),
    ("_task_utils.mmmu_mcq_utils", "parse_videommmu_multi_choice_response", "choices+index2ans", "ABCD"),
    ("FALCONBench.utils", "parse_multi_choice_response", "choices+index2ans", "ABCD"),
    ("longvideobench.utils", "parse_multi_choice_response", "choices+index2ans", "ABCDE"),
    ("egoschema.utils", "parse_multi_choice_response", "choices+index2ans", "ABCDE"),
    ("lemonade.utils", "parse_multi_choice_response", "choices+index2ans", "ABCD"),
    ("mmar.utils", "parse_multi_choice_response", "choices+index2ans", "ABCD"),
    ("mmsu.utils", "parse_multi_choice_response", "choices+index2ans", "ABCD"),
    ("nextqa.utils", "parse_multi_choice_response", "choices+index2ans", "ABCDE"),
    ("perceptioncomp.utils", "parse_multi_choice_response", "choices+index2ans", "ABCDE"),
    ("qbench.utils", "parse_multi_choice_response", "choices+index2ans", "ABCD"),
    ("video_holmes.utils", "parse_multi_choice_response", "choices+index2ans", "ABCDEF"),
    ("vmcbench.utils", "parse_multi_choice_response", "choices+index2ans", "ABCD"),
    ("air_bench.utils", "parse_multi_choice_response", "choices", "ABCD"),
    ("csbench.utils", "parse_multi_choice_response", "choices", "ABCD"),
    ("mmau.utils", "parse_multi_choice_response", "choices", "ABCD"),
    ("mmt.utils", "parse_multi_choice_response", "choices", "ABCDEF"),
    ("muchomusic.utils", "parse_multi_choice_response", "choices", "ABCD"),
    ("sitebench.utils", "parse_multi_choice_response", "choices", "ABCDEF"),
    ("super_gpqa.utils", "parse_multi_choice_response", "choices", "ABCDEF"),
    ("egoplan.utils", "extract_characters_regex", "response", "ABCD"),
    ("egoplan2.utils", "extract_characters_regex", "response", "ABCD"),
    ("longtimescope.utils", "extract_characters_regex", "response", "ABCDEF"),
    ("timescope.utils", "extract_characters_regex", "response", "ABCDEF"),
    ("longvt.utils", "extract_characters_regex", "response", "ABCD"),
    ("mme_realworld.utils", "extract_characters_regex", "response", "ABCDE"),
    ("mme_realworld.reasoning.utils", "extract_characters_regex", "response", "ABCDE"),
    ("mmworld.utils", "extract_characters_regex", "response", "ABCD"),
    ("video-tt.utils", "extract_characters_regex", "response", "ABCDE"),
    ("videomathqa.utils", "extract_characters_regex", "response", "ABCDE"),
    ("paibench_u.utils", "parse_multi_choice_response", "response", "ABCDE"),
    ("realunify.utils", "extract_answer", "response", "ABCD"),
    ("unig2u.realunify.utils", "extract_answer", "response", "ABCD"),
    ("scivideobench.utils", "extract_answer_letter", "response", "ABCDEFGHIJ"),
    ("vsibench.utils", "fuzzy_matching", "response", "ABCD"),
    ("mmvp.utils", "_extract_answer_letter", "response", "AB"),
    ("blink.utils", "_extract_answer_letter", "letters", "ABCD"),
    ("cv_bench.utils", "_extract_answer_letter", "response", "ABCDEF"),
    ("embspatial.utils", "_extract_answer_letter", "response", "ABCD"),
    ("geometry3k.utils", "_extract_answer_letter", "option_texts", "ABCD"),
    ("wm_abench.utils", "_extract_answer_letter", "response", "ABCDEF"),
    ("vstar_bench.utils", "extract_answer_letter", "response", "ABCD"),
    ("mmsi_video.utils", "extract_answer", "response", "ABCDEF"),
    ("omni_bench.utils", "parse_multi_choice_response", "choices", "ABCD"),
    ("visualwebbench.utils", "parse_multi_choice_response", "choices", "ABCDEFGH"),
    ("motionbench.utils", "_extract_choice_letter", "response", "ABCDEF"),
    ("videomme_v2.utils", "extract_characters_regex", "response", "ABCDEFGH"),
    ("lsdbench.utils", "extract_characters_regex", "response", "ABCD"),
    ("video_index.utils", "extract_letter", "option_texts", "ABCDE"),
    ("videomme.video_only_abcd.utils", "_extract_abcd", "response", "ABCD"),
    ("spatial_scene_bench.utils", "parse_answer", lambda f, response, letters: f(response, "choice", _texts(letters)), "ABCD"),
    ("seedbench.utils", "seed_process_result", _seed, "ABCD"),
    ("seedbench.ko_utils", "seed_process_result", _seed, "ABCD"),
    ("seedbench_2.utils", "seed_process_result", _seed, "ABCD"),
    ("ai2d.utils", "MultiChoiceRegexFilter", _ai2d, "ABCD"),
    ("mantis.utils", "get_prediction", lambda f, response, letters: _scored(lambda gold: f("multi-choice", response, gold, [f"({x}) {INDEX2ANS[x]}" for x in letters])["correct"], letters), "ABCD"),
    ("llava_interleave_bench.utils", "mcq_acc", lambda f, response, letters: _scored(lambda gold: f(gold, response) == 1, letters), "ABCDE"),
    ("viewspatial.utils", "viewspatial_process_results", lambda f, response, letters: _scored(lambda gold: f({"answer": gold}, [response])["overall_accuracy"]["score"] == 1.0, letters), "ABCD"),
    ("mmsi_bench.utils", "extract_single_choice_with_word_boundary", lambda f, response, letters: _scored(lambda gold: f(response, gold) == 1.0, letters), "ABCDEF"),
    ("unig2u.mmsi.utils", "extract_single_choice_with_word_boundary", lambda f, response, letters: _scored(lambda gold: f(response, gold) == 1.0, letters), "ABCD"),
    ("vinoground.utils", "vinoground_process_results", lambda f, response, letters: f({"major": "m", "minor": None, "index": "1_pos_text"}, [response])["vinoground_score"]["pred"][:1].upper(), "AB"),
    ("prismm_bench.utils", "prismm_process_results", lambda f, response, letters: _scored(lambda gold: f({"task_identification": {"answer": gold}}, [response])["exact_match"] == 1.0, letters), "ABCD"),
    ("prismm_bench.utils", "prismm_edit_process_results", lambda f, response, letters: _scored(lambda gold: f({"task_remedy": {"answer": gold}}, [response])["exact_match"] == 1.0, letters), "ABCD"),
    ("prismm_bench.utils", "prismm_pair_match_process_results", lambda f, response, letters: _scored(lambda gold: f({"task_pair_match": {"answer": gold, "letters": list(letters)}}, [response])["exact_match"] == 1.0, letters), "ABCDE"),
    ("tempcompass.utils", "tempcompass_process_results_multi_choice", _tempcompass, "ABCD"),
    ("conbench.utils", "conbench_process_results", lambda f, response, letters: _scored(lambda gold: f({"question_field": "Choices", "answer": gold, "image_id": 0}, [response])["ConScore_D"]["score"] == 1, letters), "ABCD"),
]
PARSER_IDS = [f"{module}:{function}" for module, function, _, _ in PARSERS]

# The selected letter is the first offered one that is not A (an article in English prose).
FINAL_ANSWER_FORMATS = [
    "{x}",
    "({x})",
    "{x}.",
    "{x}. {text}",
    "Answer: {x}",
    "The answer is {x}.",
    "The correct answer is **{x}**.",
    "**{x}**",
    "**Answer: {x}**",
    "\\boxed{{{x}}}",
    "<answer>{x}</answer>",
    "Option {x}",
    "I see a dog. Answer: {x}",
    "Considering the clip, the answer is {x}",
    "A dog runs across the road, so the answer is {x}.",
    "The clip shows a red car and an old book on a table.\n\nTherefore, the correct option is {x}.",
    "The clip shows a road at night.\n\nPerhaps the answer is A, but the second frame rules that out.\n\n**{x}**",
    "{x}\n\nA red car appears first, and A is the option that names it, but the question asks for the last object.",
    "The answer is {x}.\n\nThe other options do not match the last frame.",
    "The two forces are T_A = 3 N and T_C = 4 N, so the bar tilts.\n({x})",
]

# The final-answer step abstains on these; the parser's own heuristic must decide exactly as before.
UNREAD_RESPONSES = [
    "",
    "   ",
    "Because the cat is on the left.",
    "A or B",
    "A. a red car\nB. a blue bike\nC. a green tree\nD. an old book",
    "It shows a blue bike on the road near the shop.",
    "left",
    "I am not sure which one it is.",
    "Answer: A\n\nFinal answer: unknown",
    "The answer is A.\n\nThe frames are dark.\n\nOn reflection, the answer is B.",
    "42",
]


def _load(module, function):
    try:
        loaded = importlib.import_module(f"lmms_eval.tasks.{module}")
    except ImportError as error:  # an optional dependency of that task is missing
        pytest.skip(f"{module}: {error}")
    return loaded, getattr(loaded, function)


def _call(parser, style, response, letters):
    choices = list(letters)
    if callable(style):
        return style(parser, response, letters)
    if style == "letters":
        return parser(response, letters)
    if style == "option_texts":
        return parser(response, _texts(letters))
    if style == "choices+index2ans":
        return parser(response, choices, {letter: INDEX2ANS[letter] for letter in choices if letter in INDEX2ANS})
    if style == "choices":
        return parser(response, choices)
    return parser(response)


def _letter(result):
    """EgoSchema returns (letter, matched)."""
    return result[0] if isinstance(result, tuple) else result


@pytest.fixture(autouse=True)
def _fixed_random_choice(monkeypatch):
    monkeypatch.setattr(random, "choice", lambda options: options[0])


@pytest.mark.parametrize("response_format", FINAL_ANSWER_FORMATS)
@pytest.mark.parametrize(("module", "function", "style", "letters"), PARSERS, ids=PARSER_IDS)
def test_parser_reads_the_final_answer(module, function, style, letters, response_format):
    _, parser = _load(module, function)
    selected = letters[1]
    response = response_format.format(x=selected, text=INDEX2ANS[selected])
    assert _letter(_call(parser, style, response, letters)) == selected


@pytest.mark.parametrize(("module", "function", "style", "letters"), PARSERS, ids=PARSER_IDS)
def test_parser_keeps_its_own_result_when_no_final_answer_is_read(module, function, style, letters, monkeypatch):
    loaded, parser = _load(module, function)
    for response in UNREAD_RESPONSES:
        assert extract_final_mcq_answer(response, list(letters)) == "", response
    with_step = [_call(parser, style, response, letters) for response in UNREAD_RESPONSES]
    monkeypatch.setattr(loaded, "extract_final_mcq_answer", lambda *args, **kwargs: "")
    assert with_step == [_call(parser, style, response, letters) for response in UNREAD_RESPONSES]


@pytest.mark.parametrize(("module", "function", "style", "letters"), PARSERS, ids=PARSER_IDS)
def test_parser_reads_a_bare_letter_as_before(module, function, style, letters, monkeypatch):
    loaded, parser = _load(module, function)
    with_step = [_call(parser, style, letter, letters) for letter in letters]
    monkeypatch.setattr(loaded, "extract_final_mcq_answer", lambda *args, **kwargs: "")
    assert with_step == [_call(parser, style, letter, letters) for letter in letters]
    assert [_letter(result) for result in with_step] == list(letters)


# --- Scorers that compare the parsed letter themselves ---


@pytest.mark.parametrize(
    ("pred", "target", "expected"),
    [
        ("", "B", 0),  # an empty response is a substring of every target
        ("   ", "B", 0),
        ("B", "B", 1.0),
        ("b", "B", 1.0),
        ("B. the left", "B", 1.0),
        ("(B)", "B", 1.0),
        ("The answer is B", "B", 1.0),
        ("Answer: B", "B", 1.0),
        ("The camera moves left in the second view.\n\n**B**", "B", 1.0),
        ("A", "B", 0),
        ("A or B", "B", 0),
        ("Because the object is on the left.", "B", 0),  # the first letter of a word is not an option
        # Read by the scorer's own opening-letter rule, as before.
        ("B, the left one", "B", 1.0),
        ("B (the left one)", "B", 1.0),
        ("B since the chair is on the left.", "B", 1.0),
        ("B. The answer is not A.", "B", 1.0),
        ("B. left\nA. right", "B", 1.0),
        ("B\n\nOption A: wrong.", "B", 1.0),
    ],
)
def test_sparbench_exact_match(pred, target, expected):
    from lmms_eval.tasks.sparbench.utils import exact_match

    assert exact_match(pred, target) == expected


@pytest.mark.parametrize(
    ("pred", "expected"),
    [
        ("A", 1.0),
        ("a", 1.0),
        ("A. x", 1.0),
        ("(A)", 1.0),
        ("Answer: A", 1.0),
        ("The answer is A.", 1.0),
        ("**A**", 1.0),
        ("B", 0.0),
        ("B. y", 0.0),
        ("", 0.0),
        ("A or B", 0.0),
        ("C", 0.0),  # not an offered option
        # Read by the scorer's own "A. text" rule, as before.
        ("A. The correct option is the first one.", 1.0),
        ("A. The answer is not B.", 1.0),
        ("A.x", 1.0),
        ("A. x\nB. y", 1.0),
    ],
)
def test_scienceqa_process_results(pred, expected):
    from lmms_eval.tasks.scienceqa.utils import sqa_process_results

    doc = {"choices": ["x", "y"], "answer": 0}
    assert sqa_process_results(doc, [pred]) == {"exact_match": expected}


@pytest.mark.parametrize("letter", "ABCDEFGH")
@pytest.mark.parametrize("response_format", ["{x}", "({x})", "{x}.", "Answer: {x}", "**{x}**", "\\boxed{{{x}}}", "I see a dog. Answer: {x}", "A dog runs across the road, so the answer is {x}."])
def test_mvbench_scores_the_final_answer(letter, response_format):
    from lmms_eval.tasks.mvbench.utils import mcq_acc

    assert mcq_acc(letter, response_format.format(x=letter)) == 1
    other = "B" if letter != "B" else "C"
    assert mcq_acc(other, response_format.format(x=letter)) == 0


def test_mvbench_keeps_its_own_result_when_no_final_answer_is_read(monkeypatch):
    from lmms_eval.tasks.mvbench import utils as mvbench

    responses = [*UNREAD_RESPONSES, "a blue bike", "B. a blue bike", "(B) a blue bike"]
    with_step = [mvbench.mcq_acc(gold, response) for response in responses for gold in ("B", "B. a blue bike")]
    monkeypatch.setattr(mvbench, "extract_final_mcq_answer", lambda *args, **kwargs: "")
    assert with_step == [mvbench.mcq_acc(gold, response) for response in responses for gold in ("B", "B. a blue bike")]


def test_conbench_scores_an_empty_choice_response_as_wrong():
    conbench = _load("conbench.utils", "conbench_process_results")[0]
    for response in ["", "   ", "\n"]:
        assert conbench.conbench_process_results({"question_field": "Choices", "answer": "A", "image_id": 0}, [response])["ConScore_D"]["score"] == 0


def test_vinoground_scores_an_empty_response_as_wrong(monkeypatch):
    vinoground = _load("vinoground.utils", "vinoground_aggregate_results")[0]
    scores = {f"{i}_{kind}": {"GT": "A"} for i in range(500) for kind in ("pos", "neg")}
    monkeypatch.setattr(vinoground, "prep_data", lambda: (scores, scores))
    docs = [({"major": "m", "minor": None, "index": f"{i}_{kind}_{question}"}, "" if i == 0 else "A") for i in range(500) for kind in ("pos", "neg") for question in ("text", "video")]
    results = [vinoground.vinoground_process_results(doc, [response])["vinoground_score"] for doc, response in docs]
    text, video, group = vinoground.vinoground_aggregate_results(results)
    assert (text, video, group) == (99.8, 99.8, 99.8)
