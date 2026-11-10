"""Contract of ``extract_yes_no_answer``: the final yes/no of a response, or an abstention."""

import time

import pytest

from lmms_eval.tasks._task_utils.yes_no_extract import extract_yes_no_answer

REASONING = "The image shows a park with a bench. A dog sits on the bench, and a man stands next to it."


@pytest.mark.parametrize(
    ("response", "expected", "rationale"),
    [
        # Bare answers and presentation wrappers.
        ("Yes", "yes", "bare answer"),
        ("no", "no", "lowercase"),
        ("YES.", "yes", "uppercase with a full stop"),
        ("  No \n", "no", "surrounding whitespace"),
        ("**Yes**", "yes", "markdown bold"),
        ("*no*", "no", "markdown italic"),
        ("`yes`", "yes", "backticks"),
        ('"No"', "no", "quotes"),
        ("(Yes)", "yes", "parentheses"),
        ("\\text{Yes}", "yes", "LaTeX text"),
        ("\\textbf{No}.", "no", "LaTeX bold"),
        # S1: the last container decides.
        ("\\boxed{Yes}", "yes", "box"),
        ("\\boxed{\\text{No}}", "no", "box with LaTeX text"),
        ("<answer>no</answer>", "no", "answer tag"),
        ("<answer> Yes. </answer>", "yes", "answer tag with spaces and a full stop"),
        ("\\boxed{Yes, the restaurant is open.}", "yes", "box that opens with the answer"),
        (REASONING + "\n\n\\boxed{No}", "no", "box after reasoning"),
        ("\\boxed{Yes}\n\nThe dog sits on the bench, and no cat is visible.", "yes", "box before an explanation"),
        ("First guess \\boxed{No}.\n\nOn a second look, \\boxed{Yes}", "yes", "the last container wins"),
        ("\\boxed{false}", "", "a container with another word"),
        ("\\boxed{Unanswerable}", "", "a container with another word"),
        ("\\boxed{}", "", "an empty container"),
        ("Answer: No\n\n\\boxed{false}", "", "an unusable final container is not replaced by an earlier answer"),
        ("\\boxed{No. 33, Jalan Harmonium}", "", "'No.' before a number abbreviates 'number'"),
        ("Maybe \\boxed{Yes}?\n\nNo, the frames differ.\n\nSo the answer is no.", "", "a later statement of the other answer"),
        ("\\boxed{Yes}\n\nSo the answer is yes.", "yes", "a later statement of the same answer"),
        # S2: declarations.
        ("Answer: Yes", "yes", "declaration"),
        ("The answer is no.", "no", "declaration in a sentence"),
        ("**Final Answer:** No", "no", "bold declaration"),
        ("**Answer**: yes", "yes", "bold label"),
        ("The answer to the question is no.", "no", "long declaration"),
        ('The answer is "yes".', "yes", "quoted answer"),
        ("I will answer yes.", "yes", "first-person declaration"),
        ("The dog is on the bench, so the answer is yes.", "yes", "declaration at the end of a sentence"),
        (REASONING + " The answer is no, because the man holds no leash.", "no", "declaration followed by its reason"),
        ("At first the answer is yes, but the second frame shows no dog, so the answer is no.", "no", "the last declaration wins"),
        ("The answer is not yes.", "", "negated"),
        ("The answer is yes or no.", "", "not one answer"),
        ("The answer is no dog.", "", "'no' opens a phrase"),
        ("The answer is clear from the frames.", "", "no answer after the declaration"),
        ("Perhaps the answer is yes.", "", "hedged"),
        ("Maybe the answer is no.", "", "hedged"),
        ("If the dog is red, the answer is yes.", "", "conditional"),
        ("I am not sure whether the answer is yes.", "", "conditional"),
        ("Answer: yes\nThe answer is clear from the frames.", "", "an unusable last declaration abstains"),
        ("Is the answer yes?", "", "a question"),
        # S3: lines that are only yes/no.
        (REASONING + "\nYes", "yes", "answer on the last line"),
        (REASONING + "\n**No**", "no", "bold answer on the last line"),
        ("Yes\n" + REASONING, "yes", "answer on the first line"),
        ("Yes, the dog.\nNo", "", "an opening and a line disagree"),
        # S4: a block that opens with the answer.
        ("Yes, there is a dog.", "yes", "opens with yes and a comma"),
        ("No. The table is empty.", "no", "opens with no and a full stop"),
        ("No - the shop is closed.", "no", "opens with no and a dash"),
        ("Yes (the left one).", "yes", "opens with yes and a parenthesis"),
        ("No dog is visible.", "", "'No' opens a noun phrase"),
        ("No one is there.", "", "'No' opens a noun phrase"),
        ("Yes or no?", "", "not an answer"),
        ("Yes/No", "", "not an answer"),
        ("No-one", "", "not an answer"),
        ("Yes?", "", "a question"),
        ("yes it is", "", "no punctuation after the word"),
        # Never inside prose; never inferred.
        ("There is no dog.", "", "inside prose"),
        ("I know the answer.", "", "'know'"),
        ("It cannot be seen.", "", "'cannot'"),
        ("There is nothing.", "", "'nothing'"),
        ("The eyes are closed.", "", "'eyes'"),
        ("false", "", "no inference from synonyms"),
        ("True", "", "no inference from synonyms"),
        ("not visible", "", "no inference"),
        ("There is none.", "", "no inference"),
        ("y", "", "no abbreviations"),
        ("是", "", "English only"),
        # Positions: several paragraphs, the two ends.
        ("Yes\n\n" + REASONING, "yes", "answer first, reasoning after"),
        ("Answer: No\n\n" + REASONING, "no", "declaration first, reasoning after"),
        (REASONING + "\n\nTherefore, the answer is yes.", "yes", "reasoning first, answer last"),
        (REASONING + "\n\n**No**", "no", "reasoning first, bold answer last"),
        (REASONING + "\n\nThe answer is no.\n\nThe man has no leash.", "", "the last paragraph does not state; the middle is not read"),
        ("Yes.\n\n" + REASONING + "\n\nAnswer: No", "", "different answers at the two ends"),
        ("Yes.\n\n" + REASONING + "\n\nSo the answer is yes.", "yes", "the same answer at the two ends"),
        ("Yes.\n\n" + REASONING + "\n\nThe answer is clear from the frames.", "", "an unusable final declaration abstains"),
        (REASONING + "\n\nYes, I see it now; the answer is no.", "no", "the last declaration of the last paragraph"),
        ("Let me look. Yes, a dog.\n\n" + REASONING, "", "the first paragraph must open with its statement"),
        ("Yes, a dog. The answer is no.\n\n" + REASONING, "", "the first paragraph goes on to declare the other answer"),
        ("<think>" + REASONING + "</think>\n\nNo", "no", "an answer after a reasoning block"),
        # Not a string, or empty.
        (None, "", "not a string"),
        (3, "", "not a string"),
        (["yes"], "", "not a string"),
        ("", "", "empty"),
        ("  \n ", "", "blank"),
    ],
)
def test_contract(response, expected, rationale):
    assert extract_yes_no_answer(response) == expected, rationale


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (" " * 500_000 + "yes", "yes"),
        ("\n" * 500_000, ""),
        ("\n " * 250_000 + "x", ""),
        ("no " * 200_000, ""),
        ("answer " * 80_000, ""),
        ("if " * 200_000 + "the answer is yes", ""),
        ("the answer is yes " * 30_000, "yes"),
        ("yes\n" * 200_000, "yes"),
        ("\\boxed{" + "{" * 250_000, ""),
        ("\\boxed{Yes} " + "the answer is no " * 30_000, ""),
        ("\\boxed{Yes} " + "the answer is yes. " * 30_000, "yes"),
        ("\\boxed{Yes}\n" + "no\n" * 200_000, ""),
        ("the answer is" + " " * 500_000 + "yes", "yes"),
        ("I will answer " * 40_000, ""),
        ("<answer>" * 60_000, ""),
    ],
)
def test_long_degenerate_responses_are_read_in_linear_time(response, expected):
    start = time.perf_counter()
    assert extract_yes_no_answer(response) == expected
    assert time.perf_counter() - start < 2.0
