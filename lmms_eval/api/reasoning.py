import re
from typing import List, Optional, Union


def strip_reasoning_tags(text: str, tag_pairs: List[List[str]]) -> str:
    """Remove reasoning tag blocks from model output.

    Args:
        text: Raw model output string
        tag_pairs: List of [start_tag, end_tag] pairs,
                   e.g. [["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]

    Returns:
        Cleaned text with reasoning blocks removed.
    """
    result = text
    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag) :]
            else:
                break
    return result.strip()


def parse_reasoning_tags_config(cli_value: Optional[str] = None, task_value: Optional[object] = None) -> Optional[List[List[str]]]:
    """Resolve reasoning_tags from CLI + task config.

    Priority: task_value > cli_value.
    "none" / None = disabled.
    """
    import json

    effective = task_value if task_value is not None else cli_value
    if effective is None or effective == "none" or effective is False:
        return None
    if isinstance(effective, str):
        return json.loads(effective)
    return effective
