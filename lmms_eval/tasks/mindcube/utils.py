from typing import Optional
import re

def mindcube_doc_to_text(doc):
  return doc['input_prompt']

def mindcube_doc_to_visual(doc):
  return [visual.convert('RGB') for visual in doc['images']]

# This is taken directly from the official mindcube codebase
def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from model response text using regular expressions.
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
    simple_pattern_matches = list(re.finditer(r'([A-E])\.', text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)
    
    # Then check if <Answer> tag exists and extract content after it
    answer_section_match = re.search(r'<Answer>(.*?)(?:<|$)', text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        # Check for specific patterns in the answer section
        for pattern in [
            r'[Mm]y answer is ([A-E])',
            r'[Mm]y answer is ([A-E])\.',
            r'[Tt]he answer is ([A-E])',
            r'(?:Answer: )?([A-E])\.',
            r'\b([A-E])\b'
        ]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)
    
    # If no matches found after <Answer> tag, proceed with regular priority patterns
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r'(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)',  # Answer at line beginning
        r'[\*\"]([A-E])[\*\"]',  # Answer in quotes or asterisks
        r'\bAnswer:?\s*([A-E])\b',  # Answer following "Answer:"
        r'[Mm]y answer is ([A-E])',  # Added pattern for "My answer is X"
        r'[Mm]y answer is ([A-E])\.',  # Added pattern for "My answer is X."
        r'answer is ([A-E])',  # Added pattern for phrases like "The answer is X"
    ]
    
    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)
    
    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split('\n')
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
        match = re.search(r'\b([A-E])\b', line)
        if match:
            return match.group(1)
    
    return None  # No answer found

def mindcube_process_results(doc, results):
    # extract grounded answer
    grounded_output = doc['gt_answer']

    # extract predicted answer
    pred = results[0].strip()
    pred_answer = extract_answer(pred)

    score = 1.0 if pred_answer == grounded_output else 0.0

    # get type
    row_id = doc['id']
    first = row_id.split('_', 1)[0]
    type = None
    if first == 'among':
        type = 'among'
    elif first == 'rotation':
        type = 'rotation'
    elif first in ('around', 'aroundnew'):
        type = 'around'
    else:
        type = 'other'

    return {
        "overall_accuracy": {
            "score": score
        },
        "around_accuracy": {
            "score": score,
            "type": type,
        },
        "among_accuracy": {
            "score": score,
            "type": type,
        },
        "rotation_accuracy": {
            "score": score,
            "type": type,
        }
    }

def mindcube_aggregate_results(results):
  # --- Compute the total score across all results ---
    total_score = 0.0
    for res in results:
        total_score += res['score']

    # --- Compute average score safely ---
    avg_score = total_score / len(results) if results else 0.0
    return avg_score

def mindcube_aggregate_among_results(results):
    total_score = 0.0
    count = 0
    for res in results:
        if res['type'] == 'among':
            total_score += res['score']
            count += 1
    avg_score = total_score / count if count > 0 else 0.0
    return avg_score

def mindcube_aggregate_rotation_results(results):
    total_score = 0.0
    count = 0
    for res in results:
        if res['type'] == 'rotation':
            total_score += res['score']
            count += 1
    avg_score = total_score / count if count > 0 else 0.0
    return avg_score

def mindcube_aggregate_around_results(results):
    total_score = 0.0
    count = 0
    for res in results:
        if res['type'] == 'around':
            total_score += res['score']
            count += 1
    avg_score = total_score / count if count > 0 else 0.0
    return avg_score
