import pytest

from lmms_eval.tasks.auxsolidmath import utils


def score(response, answer):
    return utils.auxsolidmath_process_results({"id": "1", "answer": answer}, [response])["auxsolidmath_accuracy"]


@pytest.mark.parametrize(
    "response, answer",
    [
        ("The answer is 1", "12"),
        ("The answer is 2", "√2/2"),
        ("The answer is 3", "arccos√6/3"),
        ("Thus the distance is \\boxed{4}", "4√5/5"),
        ("The final answer is 16.", "6"),
        ("The answer is 2.5", "2"),
        ("The answer is -1/3", "1/3"),
        ("The answer is √3", "3"),
        ("The answer is 2√3", "2"),
    ],
)
def test_wrong_answer_that_shares_digits_with_the_reference_scores_zero(response, answer):
    assert score(response, answer) == 0.0


@pytest.mark.parametrize(
    "response, answer",
    [
        ("The answer is 6", "6"),
        ("The final answer is 3/4", "3/4"),
        ("The final answer is √5/5.", "√5/5"),
        ("The answer is 0.75", "3/4"),
        ("The answer is 0.5.", "0.5"),
        ("The answer is $6$", "6"),
        ("Therefore, the answer is \\boxed{60°}", "60°"),
        ("The answer is [37°, 90°].", "[37°, 90°]"),
        ("The cosine of the angle is √22/88.", "√22/88"),
    ],
)
def test_correct_answer_scores_one(response, answer):
    assert score(response, answer) == 1.0


@pytest.mark.parametrize(
    "response, extracted",
    [
        ("The final answer is 3/4", "3/4"),
        ("Final answer: 3/4", "3/4"),
        ("The answer is 0.75.", "0.75"),
        ("The answer is 2. Then we check it.", "2"),
        ("The answer is [37°, 90°].", "[37°, 90°]"),
    ],
)
def test_extract_answer_keeps_decimals_and_intervals(response, extracted):
    assert utils._extract_answer_from_response(response) == extracted


@pytest.mark.parametrize("answer", ["[37°, 90°]", "[0, 1/2]", "(3/4)^(1/4)", "[10/9, 4/3]"])
def test_number_against_an_interval_or_power_reference_does_not_crash(answer):
    assert score("The answer is 37", answer) == 0.0


def test_model_response_is_not_executed():
    assert score("\\boxed{__import__('builtins').len('abc')}", "3") == 0.0


@pytest.mark.parametrize("expression, value", [("3/4", 0.75), ("-2**0.5", -(2**0.5)), ("2*(1+0.5)", 3.0)])
def test_safe_eval_number_computes_plain_arithmetic(expression, value):
    assert utils._safe_eval_number(expression) == pytest.approx(value)


@pytest.mark.parametrize("expression", ["__import__('os').getpid()", "[37,90]", "(3/4)^(1/4)", "2(3)", "x+1", "1/0", "9**9**9", "(-8)**(1/3)"])
def test_safe_eval_number_rejects_anything_else(expression):
    assert utils._safe_eval_number(expression) is None


@pytest.mark.parametrize(
    "response",
    [
        "**Final Answer:** √6/3",
        "The final answer is **√6/3**.",
        "**Answer: √6/3**",
        "Thus, the distance is **√6/3**.",
        "The answer is **√6/3** (about 0.816).",
        "The answer is `√6/3`",
    ],
)
def test_markdown_emphasis_around_the_answer_is_ignored(response):
    assert score(response, "√6/3") == 1.0
    assert score(response, "6") == 0.0


def test_power_between_digits_is_not_markdown_emphasis():
    assert score("The answer is 2**3", "8") == 1.0
    assert score("The answer is **2**", "8") == 0.0
