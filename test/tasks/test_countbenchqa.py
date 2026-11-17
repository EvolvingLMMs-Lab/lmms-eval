import pytest

from lmms_eval.tasks.countbenchqa import utils


@pytest.mark.parametrize(
    "response, number, expected",
    [
        ("3", 3, 1.0),
        ("3.", 3, 1.0),
        (" 10.\n", 10, 1.0),
        ("There are 4 dogs.", 4, 1.0),
        ("seven", 7, 1.0),
        ("13.", 10, 0.0),
        ("I cannot tell.", 3, 0.0),
        ("", 3, 0.0),
    ],
)
def test_process_results_compares_the_parsed_count(response, number, expected):
    assert utils.countbenchqa_process_results({"number": number}, [response]) == {"acc": expected, "exact_match": expected}
