import pytest

from lmms_eval.llm_judge.base import ServerInterface
from lmms_eval.llm_judge.protocol import Request, Response
from lmms_eval.llm_judge.utils import ResponseParser


def parse(reply, output_format="0/1"):
    return ResponseParser.parse_binary_response(reply, output_format)


@pytest.mark.parametrize(
    "reply",
    [
        "0\n\nThe prediction says 12, but the reference answer is 21.",
        "0, because 2019 is not 1998",
        "0 - the model answered 1.5 while the answer is 2",
        "Score: 0 (incorrect). The prediction x = 10 does not match.",
        "0\nThe prediction is wrong; the reference mentions option 1.5",
        "[0] The answer 11 is not 12.",
        "10",
        "0.1",
        "12 is not 21",
    ],
)
def test_a_one_inside_a_longer_number_is_not_a_correct_verdict(reply):
    assert parse(reply) == 0


@pytest.mark.parametrize(
    "reply",
    [
        "1",
        " 1\n",
        "[1]",
        "**1**",
        "1.",
        "1.0",
        "Score: 1",
        "answer: 1",
        "Score: 1. The model's 12 equals the reference 12.",
        "1 - the prediction 21 matches the reference 21",
        "The prediction 1960 is right. Score: 1",
    ],
)
def test_a_standalone_one_is_a_correct_verdict(reply):
    assert parse(reply) == 1


@pytest.mark.parametrize("reply, expected", [("得分1", 1), ("1分", 1), ("输出：1", 1), ("得分0", 0), ("0分，答案是12", 0)])
def test_verdict_next_to_cjk_text_is_read(reply, expected):
    assert parse(reply) == expected


@pytest.mark.parametrize("reply", ["0", "[0]", "0.0", "Score: 0", "answer: 0", "", "incorrect", "No", "0 (not 1)", "v1"])
def test_zero_and_unreadable_replies_are_incorrect(reply):
    assert parse(reply) == 0


@pytest.mark.parametrize("output_format", ["0/1", "1/0"])
def test_both_numeric_formats_read_the_same_verdict(output_format):
    assert parse("0 - answer 12 is wrong", output_format) == 0
    assert parse("1 - answer 12 is right", output_format) == 1


@pytest.mark.parametrize("reply, expected", [("Yes", True), ("yes, it matches", True), ("No", False), ("not yes", False)])
def test_yes_no_format_is_unchanged(reply, expected):
    assert parse(reply, "yes/no") is expected


class EchoJudge(ServerInterface):
    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    def evaluate(self, request: Request) -> Response:
        return Response(content=self.reply, model_used="echo")

    def is_available(self) -> bool:
        return True


def test_evaluate_binary_scores_a_verbose_incorrect_verdict_as_incorrect():
    judge = EchoJudge("0\n\nThe prediction says 12, but the reference answer is 21.")
    result = judge.evaluate_binary(question="q", answer="21", prediction="12", output_format="0/1")
    assert result["result"] == 0


def test_evaluate_binary_scores_a_verbose_correct_verdict_as_correct():
    judge = EchoJudge("1\n\nThe prediction 21 matches the reference 21.")
    result = judge.evaluate_binary(question="q", answer="21", prediction="21", output_format="0/1")
    assert result["result"] == 1
