import re
import os
import json

from loguru import logger as eval_logger

import ast

# Compile once; case-insensitive; spans newlines
_FINAL_ANSWER_RE = re.compile(r"Final\s*answer\s*:\s*(\[[^\]]*\])", re.IGNORECASE | re.DOTALL)


def _extract_final_answer_list(response):
    """
    Strictly extract the LAST 'Final answer: [...]' list.
    Returns [] if the anchor is missing or parsing fails.
    """
    if not isinstance(response, str):
        return []

    matches = list(_FINAL_ANSWER_RE.finditer(response))
    if not matches:
        eval_logger.warning("Strict mode: 'Final answer:' anchor not found; returning empty list.")
        return []

    raw_list = matches[-1].group(1)  # last occurrence

    # Safe parse (no eval)
    parsed = None
    try:
        parsed = ast.literal_eval(raw_list)
    except Exception as e1:
        try:
            parsed = json.loads(raw_list.replace("'", '"'))
        except Exception as e2:
            eval_logger.warning(f"Strict mode: could not parse Final answer list. " f"literal_eval={e1}; json={e2}")
            return []

    if not isinstance(parsed, list):
        eval_logger.warning("Strict mode: Final answer parsed but not a list. Returning empty list.")
        return []

    # Clean + dedupe (preserve order)
    out, seen = [], set()
    for item in parsed:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def animalkingdom_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Extract visual information from the document, downloading it from HuggingFace if necessary.

    Args:
        doc (dict): A dictionary representing the document, expected to contain a "clip" key with the path or filename of the video file.
        lmms_eval_specific_kwargs (dict, optional): Additional keyword arguments specific to LMMS evaluation. Defaults to None.

    Returns:
        list[str]: A list containing the local file path to the video clip. If the file cannot be found or downloaded, returns the original path.
    """
    clip_path = doc["clip"]

    # If it's already an absolute path and exists, use it
    if os.path.isabs(clip_path) and os.path.exists(clip_path):
        return [clip_path]

    # Download from HuggingFace dataset repository
    try:
        from huggingface_hub import hf_hub_download

        # Download the video file from the HuggingFace dataset repository
        local_path = hf_hub_download(repo_id="luciehmct/animalkingdom-test", filename=clip_path, repo_type="dataset")

        if os.path.exists(local_path):
            return [local_path]
        else:
            eval_logger.error(f"Downloaded file does not exist: {local_path}")
            return [clip_path]

    except Exception as e:
        eval_logger.error(f"Failed to download video {clip_path}: {str(e)}")
        return [clip_path]


def animalkingdom_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract the prompt from the new nested dataset structure.

    Args:
        doc (dict): A dictionary representing the document with nested structure for each subtask.
        lmms_eval_specific_kwargs (dict, optional): Configuration parameters containing:
            - 'subtask': String specifying the task type ('animal', 'action', or 'activity')
            Defaults to None, which results in 'animal' subtask being used.

    Returns:
        str: The prompt for the specified animalkingdom subtask.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    subtask = lmms_eval_specific_kwargs.get("subtask", "animal")

    # DEBUG: Log the initial state
    eval_logger.debug(f"doc_to_text called with subtask from kwargs: {subtask}")
    eval_logger.debug(f"doc_to_text kwargs: {lmms_eval_specific_kwargs}")
    eval_logger.debug(f"doc_to_text - clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"doc_to_text - question_id: {doc.get('id', 'Unknown')}")

    # Detect the subtask from the call stack
    # since lmms_eval_specific_kwargs may not contain the subtask
    import inspect

    detected_subtask = subtask

    try:
        frame = inspect.currentframe()
        while frame:
            frame_locals = frame.f_locals

            # Look for task-related variables in the call stack
            if "task" in frame_locals:
                task_obj = frame_locals["task"]
                if hasattr(task_obj, "_config") and hasattr(task_obj._config, "task"):
                    task_name = task_obj._config.task
                    eval_logger.debug(f"Found task name in doc_to_text: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"
                    break

            # Additional check for task name in frame
            if "self" in frame_locals and hasattr(frame_locals["self"], "_config"):
                config = frame_locals["self"]._config
                if hasattr(config, "task"):
                    task_name = config.task
                    eval_logger.debug(f"Found task name string in doc_to_text: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"

            frame = frame.f_back
    except Exception as e:
        eval_logger.warning(f"Warning: Could not extract task from stack in doc_to_text: {e}")

    # PRIORITIZE explicit subtask from kwargs if it's valid
    if subtask in ["action", "activity", "animal"] and subtask in doc:
        detected_subtask = subtask
        eval_logger.debug(f"Using explicit subtask from kwargs: {detected_subtask}")
    else:
        eval_logger.debug(f"Using detected subtask from call stack: {detected_subtask}")

    # Use the detected subtask
    eval_logger.debug(f"Final determined subtask for prompt: {detected_subtask}")

    # Extract prompt from the nested structure
    if detected_subtask in doc and "prompt" in doc[detected_subtask]:
        prompt = doc[detected_subtask]["prompt"]
        eval_logger.debug(f"Successfully found {detected_subtask} prompt - length: {len(prompt)} chars")

        return prompt
    else:
        eval_logger.error(f"No prompt found for subtask '{detected_subtask}' in document")
        eval_logger.error(f"Available subtasks in doc: {list(doc.keys())}")
        return ""


def animalkingdom_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    """Extract target answer from the new nested dataset structure.

    Args:
        doc (dict): A dictionary representing the document with nested structure for each subtask.
        lmms_eval_specific_kwargs (dict, optional): Configuration parameters containing:
            - 'subtask': String specifying the task type ('animal', 'action', or 'activity')
            Defaults to None, which results in 'animal' subtask being used.

    Returns:
        list: The target answer for the specified animalkingdom subtask. Returns:
            - Animal target if subtask is 'animal'
            - Action target if subtask is 'action'
            - Activity target if subtask is 'activity'
            - Empty list if subtask is not recognized or no target found
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    subtask = lmms_eval_specific_kwargs.get("subtask", "animal")

    # Detect the subtask from the call stack
    # since lmms_eval_specific_kwargs may not contain the subtask
    import inspect

    detected_subtask = subtask

    try:
        frame = inspect.currentframe()
        while frame:
            frame_locals = frame.f_locals

            # Look for task-related variables in the call stack
            if "task" in frame_locals:
                task_obj = frame_locals["task"]
                if hasattr(task_obj, "_config") and hasattr(task_obj._config, "task"):
                    task_name = task_obj._config.task
                    eval_logger.debug(f"Found task name: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"
                    break

            # Additional check for task name in frame
            if "self" in frame_locals and hasattr(frame_locals["self"], "_config"):
                config = frame_locals["self"]._config
                if hasattr(config, "task"):
                    task_name = config.task
                    eval_logger.debug(f"Found task name string: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"

            frame = frame.f_back
    except Exception as e:
        eval_logger.warning(f"Warning: Could not extract task from stack in doc_to_target: {e}")

    eval_logger.debug(f"Final determined subtask: {detected_subtask}")

    # DEBUG: Log what subtask we are extracting target for
    eval_logger.debug(f"doc_to_target called with subtask: {detected_subtask}")
    eval_logger.debug(f"doc_to_target - clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"doc_to_target - question_id: {doc.get('id', 'Unknown')}")

    # Extract target from the nested structure
    if detected_subtask in doc and "answer" in doc[detected_subtask]:
        target = doc[detected_subtask]["answer"]
        eval_logger.debug(f"Returning {detected_subtask} target from nested structure: {target}")
        return target
    else:
        eval_logger.error(f"No answer found for subtask '{detected_subtask}' in document")
        return []


def animalkingdom_jaccard_metric(predictions, references):
    """Calculate the Jaccard similarity between predicted and reference labels,
    using a STRICT extractor anchored on the last 'Final answer: [...]' only.
    """
    from lmms_eval.tasks.megabench.metrics.scoring.common.metrics import jaccard_index
    import re
    import ast
    import json

    # Choose the response payload
    if isinstance(predictions, list) and len(predictions) > 0:
        response = predictions[0]
    else:
        response = predictions

    # Strict extraction from "Final answer: [...]"
    if isinstance(response, str):
        predicted_labels = []

        # Find the LAST 'Final answer: [...]' (case-insensitive, spans newlines)
        matches = list(re.finditer(r"Final\s*answer\s*:\s*(\[[^\]]*\])", response, re.IGNORECASE | re.DOTALL))
        if not matches:
            eval_logger.error("Strict mode: 'Final answer:' anchor not found; using empty list.")
        else:
            raw_list = matches[-1].group(1)
            try:
                predicted_labels = ast.literal_eval(raw_list)  # safe parse
            except Exception as e1:
                # JSON-like fallback (single → double quotes)
                try:
                    predicted_labels = json.loads(raw_list.replace("'", '"'))
                except Exception as e2:
                    eval_logger.error(f"Strict mode: could not parse Final answer list. " f"literal_eval={e1}; json={e2}")
                    predicted_labels = []

            if not isinstance(predicted_labels, list):
                eval_logger.error("Strict mode: Final answer parsed but not a list; using empty list.")
                predicted_labels = []
            else:
                # Clean & dedupe while preserving order
                seen, cleaned = set(), []
                for item in predicted_labels:
                    s = str(item).strip()
                    if s and s not in seen:
                        seen.add(s)
                        cleaned.append(s)
                predicted_labels = cleaned

        parsed_prediction = predicted_labels
    else:
        # If predictions is already a list (pre-parsed), use it directly
        parsed_prediction = predictions if isinstance(predictions, list) else [predictions]

    # Normalize references to a list
    if isinstance(references, str):
        references = [references]

    # Calculate jaccard index
    jaccard_score = jaccard_index(parsed_prediction, references)
    eval_logger.debug(f"Jaccard calculation: pred={parsed_prediction}, ref={references}, score={jaccard_score}")

    return {"animalkingdom_jaccard": jaccard_score}


def animalkingdom_process_results(doc, results, lmms_eval_specific_kwargs=None):
    """Process model results to extract labels strictly from 'Final answer: [...]'."""
    if not results or len(results) == 0:
        eval_logger.warning("No results provided to process_results")
        return []

    response = results[0].strip()
    subtask = lmms_eval_specific_kwargs.get("subtask", "animal") if lmms_eval_specific_kwargs else "animal"

    # DEBUG: Log clip_path and question_id for extraction from logs
    eval_logger.debug(f"process_results - clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"process_results - full_model_response: {repr(response)}")

    # Try to detect the subtask from the call stack
    import inspect

    detected_subtask = subtask
    try:
        frame = inspect.currentframe()
        while frame:
            frame_locals = frame.f_locals

            # Look for task-related variables in the call stack
            if "task" in frame_locals:
                task_obj = frame_locals["task"]
                if hasattr(task_obj, "_config") and hasattr(task_obj._config, "task"):
                    task_name = task_obj._config.task
                    eval_logger.debug(f"Found task name in process_results: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"
                    break

            # Additional check for task name in frame
            if "self" in frame_locals and hasattr(frame_locals["self"], "_config"):
                config = frame_locals["self"]._config
                if hasattr(config, "task"):
                    task_name = config.task
                    eval_logger.debug(f"Found task name string in process_results: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"

            frame = frame.f_back
    except Exception as e:
        eval_logger.warning(f"Warning: Could not extract task from stack in process_results: {e}")

    # Use the detected subtask
    subtask = detected_subtask

    # DEBUG: Log the actual model response with clip and question info
    eval_logger.debug(f"Processing model response for subtask: {subtask}")
    eval_logger.debug(f"clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"question_id: {doc.get('id', 'Unknown')}")
    eval_logger.debug(f"Full response: {repr(response)}")

    # ==== STRICT extraction: only accept the LAST "Final answer: [...]" ====
    import re, ast, json

    predicted_labels = []

    matches = list(re.finditer(r"Final\s*answer\s*:\s*(\[[^\]]*\])", response, re.IGNORECASE | re.DOTALL))
    if not matches:
        eval_logger.error("Strict mode: No valid 'Final answer:' found; returning empty list.")
    else:
        raw_list = matches[-1].group(1)  # last occurrence
        try:
            predicted_labels = ast.literal_eval(raw_list)  # safe parse
        except Exception as e1:
            # JSON-like fallback (single → double quotes)
            try:
                predicted_labels = json.loads(raw_list.replace("'", '"'))
            except Exception as e2:
                eval_logger.error(f"Strict mode: could not parse Final answer list. " f"literal_eval={e1}; json={e2}")
                predicted_labels = []

        if not isinstance(predicted_labels, list):
            eval_logger.error("Strict mode: Final answer parsed but not a list; using empty list.")
            predicted_labels = []
        else:
            # Clean & dedupe while preserving order
            seen, cleaned = set(), []
            for item in predicted_labels:
                s = str(item).strip()
                if s and s not in seen:
                    seen.add(s)
                    cleaned.append(s)
            predicted_labels = cleaned

    eval_logger.debug(f"Final extracted labels: {predicted_labels}")

    # Compute jaccard score directly since we have both prediction and target
    def jaccard_index(pred_set, target_set):
        """Calculate Jaccard index between two sets"""
        pred_set = set(pred_set) if pred_set else set()
        target_set = set(target_set) if target_set else set()

        if len(pred_set) == 0 and len(target_set) == 0:
            return 1.0

        intersection = len(pred_set.intersection(target_set))
        union = len(pred_set.union(target_set))

        return intersection / union if union > 0 else 0.0

    # Get the target for comparison using nested structure
    target_labels = []
    if subtask in doc and "answer" in doc[subtask]:
        target_labels = doc[subtask]["answer"]
        eval_logger.debug(f"Extracted target from nested structure: {target_labels}")
    else:
        eval_logger.error(f"No answer found for subtask '{subtask}' in document structure")
        target_labels = []

    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Calculate jaccard score
    jaccard_score = jaccard_index(predicted_labels, target_labels)
    eval_logger.info(f"INFO: Jaccard score: pred={predicted_labels}, target={target_labels}, score={jaccard_score}")
    eval_logger.debug(f"INFO: Final result for clip_path={doc.get('clip', 'Unknown')}, question_id={doc.get('id', 'Unknown')}, subtask={subtask}")

    # Save comprehensive results to subtask-specific JSONL files
    try:
        doc_id = doc.get("id", "Unknown")
        clip_path = doc.get("clip", "Unknown")

        # Get the enhanced prompt with frame timestamps from InternVL3's cache
        original_prompt = ""
        enhanced_prompt = ""
        try:
            # InternVL3 saves enhanced prompts to animalkingdom_enhanced_prompts.json
            prompt_cache_file = "animalkingdom_enhanced_prompts.json"
            if os.path.exists(prompt_cache_file):
                with open(prompt_cache_file, "r", encoding="utf-8") as f:
                    prompt_cache = json.load(f)

                doc_id_str = str(doc.get("id", "Unknown"))

                # Only use subtask-specific cache; warn if missing
                if doc_id_str in prompt_cache:
                    cached_entry = prompt_cache[doc_id_str]
                    if isinstance(cached_entry, dict) and subtask in cached_entry:
                        enhanced_prompt = cached_entry[subtask].get("enhanced_prompt_with_timestamps", "")
                        original_prompt = cached_entry[subtask].get("original_prompt", "")
                        eval_logger.debug(f"Retrieved enhanced {subtask} prompt from subtask-specific cache for doc_id: {doc_id_str}")
                        eval_logger.debug(f"Enhanced prompt preview: {enhanced_prompt[:200]}...")
                    else:
                        eval_logger.warning(f"No enhanced prompt for subtask '{subtask}' in cache for doc_id: {doc_id_str}. Cache entry keys: {list(cached_entry.keys()) if isinstance(cached_entry, dict) else 'not a dict'}")
                else:
                    eval_logger.debug(f"No cached enhanced prompt found for doc_id: {doc_id_str}")
            else:
                eval_logger.debug(f"Enhanced prompt cache file not found: {prompt_cache_file}")
        except Exception as e:
            eval_logger.debug(f"Could not load InternVL3 enhanced prompt cache: {e}")

        # Fallback to original prompt from document if no enhanced prompt available
        if not enhanced_prompt and subtask in doc and "prompt" in doc[subtask]:
            original_prompt = doc[subtask]["prompt"]
            enhanced_prompt = original_prompt
            eval_logger.warning(f"No enhanced prompt with timestamps found for subtask {subtask}, doc_id: {doc_id_str}")
            eval_logger.warning("Using original prompt as fallback - timestamps may be missing")
            eval_logger.debug(f"Fallback prompt preview: {enhanced_prompt[:200]}...")
        elif not enhanced_prompt:
            eval_logger.error(f"Could not find any prompt for subtask {subtask}, doc_id: {doc_id_str}")
            enhanced_prompt = "ERROR: No prompt found"

        # Log what prompt is being saved to JSONL
        if enhanced_prompt and enhanced_prompt != "ERROR: No prompt found":
            if "<video>" in enhanced_prompt:
                eval_logger.warning(f"Saving ORIGINAL prompt (contains <video>) to JSONL for subtask {subtask}, doc_id: {doc_id_str}")
            else:
                eval_logger.info(f"Saving ENHANCED prompt (with timestamps) to JSONL for subtask {subtask}, doc_id: {doc_id_str}")
            eval_logger.debug(f"Prompt being saved: {enhanced_prompt[:200]}...")

        # Create comprehensive result entry - with jaccard_score included
        comprehensive_result = {
            "id": doc_id,
            "clip": clip_path,
            "prompt": enhanced_prompt,  # Use enhanced prompt with frame timestamps
            "full_answer": response,
            "answer": predicted_labels,
            "ground_truth": target_labels,
            "jaccard_score": jaccard_score,
        }

        # Create model_used_date_time directory structure in results directory
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_name = "InternVL3-8B"  # Can be made configurable if needed

        # Use results directory in the lmms-eval repository
        results_base_dir = os.path.join(os.getcwd(), "results")
        output_dir = os.path.join(results_base_dir, f"{model_name}_{timestamp}")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save to subtask-specific JSONL file in the structured directory
        subtask_file = os.path.join(output_dir, f"animalkingdom_{subtask}.jsonl")

        # Append to the subtask-specific file
        with open(subtask_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(comprehensive_result, ensure_ascii=False) + "\n")

        eval_logger.debug(f"Saved comprehensive result to {subtask_file}: {doc_id}")

    except Exception as e:
        eval_logger.warning(f"Failed to save comprehensive result: {e}")

    # Return the computed jaccard score
    return {"jaccard": jaccard_score}


# Aggregation function for animalkingdom jaccard results
def animalkingdom_jaccard_aggregation(items):
    """Aggregate Jaccard scores by averaging them.

    Args:
        items (list): A list of Jaccard score dictionaries from individual evaluations.

    Returns:
        float: The average Jaccard score.
    """
    if not items:
        return 0.0

    def jaccard_index(pred_set, target_set):
        """Calculate Jaccard index between two sets"""
        pred_set = set(pred_set) if pred_set else set()
        target_set = set(target_set) if target_set else set()

        if len(pred_set) == 0 and len(target_set) == 0:
            return 1.0

        intersection = len(pred_set.intersection(target_set))
        union = len(pred_set.union(target_set))

        return intersection / union if union > 0 else 0.0

    # The aggregation function receives the results from the metric computation
    # Each item should be the jaccard score computed by the framework
    total_jaccard = 0.0
    count = 0

    for item in items:
        if isinstance(item, (int, float)):
            # This is a computed jaccard score
            total_jaccard += item
            count += 1
        elif isinstance(item, list):
            # This should not happen with the proper setup, but handle it anyway
            eval_logger.warning(f"Warning: Aggregation received raw list: {item}")
        else:
            eval_logger.warning(f"Warning: Unexpected item type in aggregation: {type(item)}, value: {item}")

    return total_jaccard / count if count > 0 else 0.0
