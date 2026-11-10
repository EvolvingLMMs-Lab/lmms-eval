"""Native option-letter parsers that read the final answer first and keep their own heuristic as the fallback.

Three properties are checked for every parser:

* a declared final answer in a common format is read;
* a response the final-answer step does not read is parsed exactly as before;
* a bare option letter gives the same result as before.
"""

import importlib
import random

import pytest

from lmms_eval.tasks._task_utils.mcq_extract import extract_final_mcq_answer

INDEX2ANS = {"A": "a red car", "B": "a blue bike", "C": "a green tree", "D": "an old book", "E": "a tall tower", "F": "a small boat"}

# (module under lmms_eval.tasks, function, call style, offered letters)
#   "choices+index2ans": f(response, all_choices, index2ans)
#   "choices":           f(response, all_choices)
#   "response":          f(response)
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
        ("1", "12", 0),
        ("B", "B", 1.0),
        ("b", "B", 1.0),
        ("B. the left", "B", 1.0),
        ("(B)", "B", 1.0),
        ("The answer is B", "B", 1.0),
        ("Answer: B", "B", 1.0),
        ("The camera moves left in the second view.\n\n**B**", "B", 1.0),
        ("A", "B", 0),
        ("A or B", "B", 0),
        ("Because the object is on the left.", "B", 0),
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
