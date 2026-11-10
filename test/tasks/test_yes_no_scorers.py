"""Yes/no scorers that read the stated final answer first and keep their own heuristic as the fallback.

Three properties are checked for every patched scorer:

* a final answer in a common format is read;
* a response that ``extract_yes_no_answer`` does not read is scored exactly as before;
* a bare ``yes``/``no`` gives the same result (value and type) as before wherever the old scorer read it.

"Before" is the same scorer with the module attribute ``extract_yes_no_answer`` replaced by a reader that
always abstains, which leaves only the scorer's own heuristic.
"""

import collections
import importlib

import pytest

from lmms_eval.tasks._task_utils.yes_no_extract import extract_yes_no_answer


def _flag(gold):
    return 1 if gold == "yes" else 0


def _pope(module, response, gold):
    return module.pope_process_results({"answer": gold, "question_id": 0}, [response])


def _conbench(module, response, gold):
    return module.conbench_process_results({"question_field": "N/Y", "answer": gold.capitalize(), "image_id": 0}, [response])


def _hallusion(module, response, gold):
    record = {"model_prediction": response, "gt_answer": str(_flag(gold)), "category": "VD", "subcategory": "chart", "set_id": 0, "figure_id": 0, "question_id": 0}
    return module.hb_aggregation_result_intern([record], "aAcc")


def _amber(module, response, gold):
    metrics = collections.defaultdict(int)
    module.process_discriminative_task({"question_id": 0, "text": response}, {"type": "discriminative-hallucination", "truth": gold}, metrics)
    return dict(metrics)


def _mvh(module, response, gold):
    doc = {"question_index": 2, "binary_answers": [gold.capitalize(), "No"], "group_id": 0, "mvh_type": "object"}
    return module.mvh_process_results(doc, [response])


def _slake(module, response, gold):
    doc = {"answer": gold.capitalize(), "qid": 1, "q_lang": "en", "answer_type": "CLOSED", "modality": "CT", "content_type": "Organ"}
    return module.slake_process_results(doc, [response])


def _vlms_are_biased(module, response, gold):
    other = "No" if gold == "yes" else "Yes"
    return module.vlms_are_biased_process_results({"ground_truth": "{%s}" % gold.capitalize(), "expected_bias": "{%s}" % other, "topic": "logo"}, [response])


def _wm_abench(module, response, gold):
    return module.wm_abench_process_results({"answer": gold}, [response])


def _ovobench(module, response, gold):
    _, scores = module.calculate_score_forward([{"task": "SSR", "test_info": [{"response": response, "type": _flag(gold)}]}])
    return scores


# (id, module under lmms_eval.tasks, call(module, response, gold) -> output, correct(output, gold) -> bool)
SITES = [
    ("pope", "pope.utils", _pope, lambda out, gold: out["pope_accuracy"]["score"] == 1.0),
    ("mme", "mme.utils", lambda m, r, g: m.parse_pred_ans(r), lambda out, gold: out == gold),
    ("conbench", "conbench.utils", _conbench, lambda out, gold: out["ConScore_D"]["score"] == 1),
    ("hallusion_bench_intern", "hallusion_bench.evaluate_hb", _hallusion, lambda out, gold: out == 1.0),
    ("amber_g", "amber_g.utils", _amber, lambda out, gold: out.get("qa_correct_score", 0) == 1),
    ("robo_spatial", "robo_spatial.utils", lambda m, r, g: m.evaluate_binary_answer(g, r), lambda out, gold: out is True),
    ("vlmsareblind", "vlmsareblind.utils", lambda m, r, g: m.parse_response(r, "Touching Circles"), lambda out, gold: out == gold),
    ("naturalbench", "naturalbench.utils", lambda m, r, g: m.extract_answer(r, "yes_no"), lambda out, gold: out == _flag(gold)),
    ("camerabench_vqa", "camerabench_vqa.utils", lambda m, r, g: m.extract_answer(r, "yes_no"), lambda out, gold: out == _flag(gold)),
    ("timeblind", "timeblind.utils", lambda m, r, g: m.extract_answer(r, "yes_no"), lambda out, gold: out == _flag(gold)),
    ("tempcompass", "tempcompass.utils", lambda m, r, g: m.extract_pred(r), lambda out, gold: out == gold),
    ("spatialtreebench", "spatialtreebench.metrics.rule_metrics", lambda m, r, g: m.judgemodel(r, g.capitalize(), None), lambda out, gold: out["score"] == 1),
    ("videonet", "videonet.binary_utils", lambda m, r, g: m._extract_binary_prediction(r), lambda out, gold: out == gold),
    ("mmlongbench", "mmlongbench.utils", lambda m, r, g: m._extract_binary_label(r), lambda out, gold: out == gold),
    ("mvh_bench", "mvh_bench.utils", _mvh, lambda out, gold: out["binary_acc"]["correct"] is True),
    ("slake", "slake.utils", _slake, lambda out, gold: out["slake_accuracy"]["is_correct"] is True),
    ("corecognition", "corecognition.utils", lambda m, r, g: m._template_match(r, "YORN"), lambda out, gold: out == gold.upper()),
    ("spatial_scene_bench", "spatial_scene_bench.utils", lambda m, r, g: m.parse_answer(r, "yesno"), lambda out, gold: out == gold),
    ("vlms_are_biased", "vlms_are_biased.utils", _vlms_are_biased, lambda out, gold: out["accuracy"] == 1.0),
    ("wm_abench", "wm_abench.utils", _wm_abench, lambda out, gold: out["wm_abench_acc"]["is_correct"] is True),
    ("ovobench_forward", "ovobench.score_utils.score", _ovobench, lambda out, gold: out["SSR"] == [1]),
]
SITE_IDS = [site[0] for site in SITES]

# {Y}/{y}: the gold answer capitalized/lowercase. The explanations hold the traps of the old heuristics
# ("know", "nothing", "eyes", "not", a "no" or "yes" in the prose).
FINAL_ANSWER_FORMATS = [
    "{y}",
    "{Y}",
    "{Y}.",
    "**{Y}**",
    "`{y}`",
    "\\boxed{{{Y}}}",
    "\\boxed{{\\text{{{Y}}}}}",
    "<answer>{Y}</answer>",
    "Answer: {Y}",
    "**Answer:** {Y}",
    "Final answer: {y}",
    "The answer is {y}.",
    "{Y}, I know the eyes of the dog are not closed.",
    "{Y}. There is nothing on the left, and no cat; yes, a dog.",
    "The eyes of the dog are not visible, and I know nothing more, so the answer is {y}.",
    "The image shows a dog with its eyes closed.\n{Y}",
    "The image shows a dog with its eyes closed. Is there a cat? Nothing suggests one.\n\n**{Y}**",
    "{Y}\n\nThe image shows a dog. I know there is no cat and nothing else; yes, the eyes are closed.",
    "Let me think. Perhaps the answer is no, perhaps yes.\n\nThe second frame settles it.\n\nTherefore, the answer is {y}.",
]

# The reader abstains on these; each scorer must decide exactly as before.
UNREAD_RESPONSES = [
    "",
    "   ",
    "There is no dog in the image.",
    "I know the answer.",
    "The eyes are closed.",
    "nothing",
    "false",
    "true",
    "y",
    "not visible",
    "Yes or no?",
    "Yes/No",
    "No one is there.",
    "yes it is",
    "No dog is visible, yes.",
    "Perhaps the answer is yes.",
    "Answer: Yes\n\nThe frames are dark.\n\nAnswer: No",
    "The answer is yes.\n\nThe answer is clear from the frames.",
    "\\boxed{false}",
    "\\boxed{}",
    "是",
    "Yes, the dog.\nNo",
]

BARE_ANSWERS = ["yes", "no", "Yes", "No", "YES", "NO", "Yes.", "No.", "yes.", "no.", " Yes ", "No\n", "y", "n"]


def _load(module):
    try:
        return importlib.import_module(f"lmms_eval.tasks.{module}")
    except ImportError as error:  # an optional dependency of that task is missing
        pytest.skip(f"{module}: {error}")


def _outcome(call, module, response, gold):
    """The scorer output, or the type of the exception it raises (videonet raises on an empty response)."""
    try:
        return call(module, response, gold)
    except Exception as error:  # the old and the new scorer must fail in the same way
        return f"raises {type(error).__name__}"


def _old(module, monkeypatch):
    monkeypatch.setattr(module, "extract_yes_no_answer", lambda response: "")


@pytest.mark.parametrize("response_format", FINAL_ANSWER_FORMATS)
@pytest.mark.parametrize(("site", "module", "call", "correct"), SITES, ids=SITE_IDS)
def test_scorer_reads_the_final_answer(site, module, call, correct, response_format):
    loaded = _load(module)
    for gold, other in (("yes", "no"), ("no", "yes")):
        response = response_format.format(y=gold, Y=gold.capitalize())
        assert correct(call(loaded, response, gold), gold), (site, response)
        assert not correct(call(loaded, response, other), other), (site, response)


@pytest.mark.parametrize(("site", "module", "call", "correct"), SITES, ids=SITE_IDS)
def test_scorer_keeps_its_own_result_when_no_final_answer_is_read(site, module, call, correct, monkeypatch):
    loaded = _load(module)
    for response in UNREAD_RESPONSES:
        assert extract_yes_no_answer(response) == "", response
    cases = [(response, gold) for response in UNREAD_RESPONSES for gold in ("yes", "no")]
    with_reader = [_outcome(call, loaded, response, gold) for response, gold in cases]
    _old(loaded, monkeypatch)
    assert with_reader == [_outcome(call, loaded, response, gold) for response, gold in cases]


@pytest.mark.parametrize(("site", "module", "call", "correct"), SITES, ids=SITE_IDS)
def test_scorer_reads_a_bare_answer_as_before(site, module, call, correct, monkeypatch):
    loaded = _load(module)
    cases = [(response, gold) for response in BARE_ANSWERS for gold in ("yes", "no")]
    with_reader = [_outcome(call, loaded, response, gold) for response, gold in cases]
    _old(loaded, monkeypatch)
    before = [_outcome(call, loaded, response, gold) for response, gold in cases]
    for (response, gold), new, old in zip(cases, with_reader, before):
        if isinstance(old, str) and old.startswith("raises"):
            continue
        if correct(old, gold):
            assert type(new) is type(old) and new == old, (site, response, gold, old, new)
        if response.strip(" .\n").lower() in ("yes", "no"):
            assert correct(new, gold) == (response.strip(" .\n").lower() == gold), (site, response, gold, new)
