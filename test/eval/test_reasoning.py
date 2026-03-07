from lmms_eval.api.reasoning import strip_reasoning_tags


def test_strip_reasoning_tags_removes_paired_block():
    text = "<think>\nreasoning\n</think>\n\nYes"
    cleaned = strip_reasoning_tags(text, [["<think>", "</think>"]])
    assert cleaned == "Yes"


def test_strip_reasoning_tags_handles_prompt_prefilled_opening_tag():
    text = "reasoning from completion only\n</think>\n\nNo"
    cleaned = strip_reasoning_tags(text, [["<think>", "</think>"]])
    assert cleaned == "No"
