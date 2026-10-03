from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer


def test_standard_formats_still_parse():
    assert extract_mcq_answer("B") == "B"
    assert extract_mcq_answer("The correct answer is (B).") == "B"
    assert extract_mcq_answer("Answer: A") == "A"
    assert extract_mcq_answer("I pick C and stop") == "C"
    assert extract_mcq_answer("The best option is B)") == "B"
    assert extract_mcq_answer("A. because it handles text better", ["A", "B"]) == "A"
    assert extract_mcq_answer("(A) because it handles text better", ["A", "B"]) == "A"
    assert extract_mcq_answer("Option A is correct") == "A"


def test_lowercase_outputs_parse():
    assert extract_mcq_answer("the answer is b") == "B"
    assert extract_mcq_answer("(b) the nucleus") == "B"
    assert extract_mcq_answer("b. the nucleus") == "B"
    assert extract_mcq_answer("the answer is a") == "A"
    assert extract_mcq_answer("b is correct") == "B"
    assert extract_mcq_answer("b") == "B"
    assert extract_mcq_answer("a") == "A"
    assert extract_mcq_answer("<answer> c") == "C"
    assert extract_mcq_answer("answer> d") == "D"


def test_answer_phrase_prefers_uppercase_over_trailing_prose():
    # "and" contains a lowercase "d" that must not outweigh the uppercase "C"
    assert extract_mcq_answer("I pick C and stop") == "C"


def test_prose_is_not_mistaken_for_answers():
    assert extract_mcq_answer("A triangle has three sides.") == ""
    assert extract_mcq_answer("The DNA helix unwinds during replication.") == ""
    assert extract_mcq_answer("I saw a movie") == ""
    assert extract_mcq_answer("It costs 5 a unit") == ""
    assert extract_mcq_answer("the value of c") == ""
    assert extract_mcq_answer("I'll go with plan b") == ""
    assert extract_mcq_answer("no letters here") == ""


def test_empty_response_returns_empty():
    assert extract_mcq_answer("") == ""
    assert extract_mcq_answer("   ") == ""
