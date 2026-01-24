import json
import re
from typing import Dict, Optional, Tuple


def get_setting_from_id(item_id: str) -> str:
    """Extract setting type from item ID.

    Args:
        item_id: The item identifier

    Returns:
        Setting type ('around', 'rotation', 'translation', 'among', 'other')

    """
    if not item_id:
        return "other"

    item_id_lower = item_id.lower()

    if "around" in item_id_lower:
        return "around"
    elif "rotation" in item_id_lower:
        return "rotation"
    elif "translation" in item_id_lower:
        return "translation"
    elif "among" in item_id_lower:
        return "among"
    else:
        return "other"


def determine_answer_fields(item: Dict) -> Tuple[str, str]:
    """Determine which fields contain the answers.

    Args:
        item: The evaluation item dictionary

    Returns:
        Tuple of (cogmap_field, plain_field)

    """
    # Check for different field names that might be used
    if "cogmap_gen_answer" in item:
        cogmap_field = "cogmap_gen_answer"
    elif "cogmap_answer" in item:
        cogmap_field = "cogmap_answer"
    else:
        cogmap_field = "answer"  # Default

    # For plain answer field
    if "plain_answer" in item:
        plain_field = "plain_answer"
    else:
        plain_field = "answer"  # Default

    return cogmap_field, plain_field


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer from model response text using regular expressions.
    Returns the last occurrence of the letter of the answer (A, B, C, D, or E)
    based on pattern priority - tries higher priority patterns first.

    Args:
        text: The model response text

    Returns:
        The last answer letter found by the highest priority matching pattern,
        or None if not found

    """
    if not text:
        return None

    # First, try to match simple answer format: A., B., C., D., E. with highest priority
    simple_pattern_matches = list(re.finditer(r"([A-E])\.", text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)

    # Then check if <Answer> tag exists and extract content after it
    answer_section_match = re.search(r"<Answer>(.*?)(?:<|$)", text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        # Check for specific patterns in the answer section
        for pattern in [
            r"[Mm]y answer is ([A-E])",
            r"[Mm]y answer is ([A-E])\.",
            r"[Tt]he answer is ([A-E])",
            r"(?:Answer: )?([A-E])\.",
            r"\b([A-E])\b",
        ]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)

    # If no matches found after <Answer> tag, proceed with regular priority patterns
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r"(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)",  # Answer at line beginning
        r"[\*\"]([A-E])[\*\"]",  # Answer in quotes or asterisks
        r"\bAnswer:?\s*([A-E])\b",  # Answer following "Answer:"
        r"[Mm]y answer is ([A-E])",  # Added pattern for "My answer is X"
        r"[Mm]y answer is ([A-E])\.",  # Added pattern for "My answer is X."
        r"answer is ([A-E])",  # Added pattern for phrases like "The answer is X"
    ]

    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)

    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split("\n")
    line_matches = []

    for i, line in enumerate(lines):
        # Look for full answer pattern in each line
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))

    if line_matches:
        # Return the answer from the last line that matched
        return line_matches[-1][1]

    # Finally, try the most general pattern on each line
    for i in reversed(range(len(lines))):  # Start from bottom
        line = lines[i]
        match = re.search(r"\b([A-E])\b", line)
        if match:
            return match.group(1)

    return None  # No answer found


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON cognitive map from text response.
    Returns the JSON object if found, otherwise None.

    Args:
        text: Text containing a JSON object

    Returns:
        Extracted JSON object or None

    """
    if not text:
        return None

    # Look for JSON pattern with { } brackets
    pattern = r"\{[\s\S]*\}"
    matches = re.findall(pattern, text)

    if not matches:
        return None

    # If multiple matches, select the longest one
    matches.sort(key=len, reverse=True)
    json_str = matches[0]

    # Try direct JSON parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to clean up and parse again
        return clean_and_parse_json(json_str)


def clean_and_parse_json(json_str: str) -> Optional[Dict]:
    """Attempt to clean and parse a malformed JSON string.

    Args:
        json_str: A potentially malformed JSON string

    Returns:
        Parsed JSON object or None

    """
    try:
        # Remove comments
        clean_json = re.sub(r"//.*", "", json_str)
        # Remove newlines, tabs
        clean_json = re.sub(r"[\n\r\t]", " ", clean_json)

        # Fix unquoted keys
        clean_json = re.sub(r"(\s*?)(\w+)(\s*?):", r'\1"\2"\3:', clean_json)
        # Fix trailing commas
        clean_json = re.sub(r",\s*}", "}", clean_json)
        clean_json = re.sub(r",\s*]", "]", clean_json)

        return json.loads(clean_json)
    except Exception:
        # As a final attempt, try to extract in "key-value" format
        try:
            # Extract pairs like "object_name": { "position": [...], "facing": ... }
            pairs_pattern = r'"([^"]+)":\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
            pairs = re.findall(pairs_pattern, json_str)

            if pairs:
                result = {}
                for key, value in pairs:
                    try:
                        # Parse the value part
                        value_str = "{" + value + "}"
                        # Fix unquoted keys
                        value_str = re.sub(r"(\s*?)(\w+)(\s*?):", r'\1"\2"\3:', value_str)
                        # Fix trailing commas
                        value_str = re.sub(r",\s*}", "}", value_str)
                        value_str = re.sub(r",\s*]", "]", value_str)

                        value_obj = json.loads(value_str)
                        result[key] = value_obj
                    except Exception:
                        continue

                if result:
                    return result
        except Exception:
            pass

        return None


def _extract_cognitive_map(cogmap_answer: str) -> Optional[Dict]:
    """Extract cognitive map from response with error handling."""
    try:
        # First try direct extraction from the answer text
        generated_cogmap = extract_json_from_text(cogmap_answer)

        # # If extraction failed, check for separately stored cognitive map
        # if not generated_cogmap:
        #     if 'cognitive_map' in item:
        #         cognitive_map = item.get('cognitive_map')
        #         if isinstance(cognitive_map, str):
        #             generated_cogmap = extract_json_from_text(cognitive_map)
        #         else:
        #             generated_cogmap = cognitive_map
        #     elif cogmap_field + '_cognitive_map' in item:
        #         cognitive_map = item.get(cogmap_field + '_cognitive_map')
        #         if isinstance(cognitive_map, str):
        #             generated_cogmap = extract_json_from_text(cognitive_map)
        #         else:
        #             generated_cogmap = cognitive_map
        #     elif 'grounded_cogmap' in item and item['grounded_cogmap'] is not None:
        #         if cogmap_field.startswith('gen') or cogmap_field.startswith('cogmap_gen'):
        #             cognitive_map = item.get('grounded_cogmap')
        #             if isinstance(cognitive_map, str):
        #                 generated_cogmap = extract_json_from_text(cognitive_map)
        #             else:
        #                 generated_cogmap = cognitive_map

        return generated_cogmap

    except Exception:
        return None


def _extract_grounded_cogmap(grounded_cogmap: str) -> Optional[Dict]:
    """Extract grounded cognitive map from item."""
    if isinstance(grounded_cogmap, str):
        grounded_cogmap = extract_json_from_text(grounded_cogmap)
    return grounded_cogmap
