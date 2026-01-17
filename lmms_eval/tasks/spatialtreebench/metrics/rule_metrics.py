import ast
import json
import os
import random
import re

import numpy as np
from scipy.spatial.transform import Rotation as R


def gravityeval(response, answer, extra_info):
    """
    Calculates the score for the gravity metric.
    Args:
        response: The response from the model.
        answer: The answer to the question.
        extra_info: The gravity metric must contain the following keys: roll_unc, pitch_unc, vfov_unc.
    Returns:
        A dictionary containing the score and the details.
    """
    response_str = ""
    match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if match:
        response_str = match.group(1)
    else:
        match = re.search(r"```\n(.*?)\n```", response, re.DOTALL)
        if match:
            response_str = match.group(1)
        else:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and start < end:
                response_str = response[start : end + 1]
            else:
                roll_match = re.search(r'["\']?roll["\']?\s*[:=]\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
                pitch_match = re.search(r'["\']?pitch["\']?\s*[:=]\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
                vfov_match = re.search(r'["\']?vfov["\']?\s*[:=]\s*(-?\d+\.?\d*)', response, re.IGNORECASE)

                if roll_match and pitch_match and vfov_match:
                    roll = roll_match.group(1)
                    pitch = pitch_match.group(1)
                    vfov = vfov_match.group(1)
                    response_str = f'{{"roll": {roll}, "pitch": {pitch}, "vfov": {vfov}}}'
                else:
                    response_str = response

    try:
        pred_data = json.loads(response_str)
    except json.JSONDecodeError:
        try:
            response_str_fixed = re.sub(r",\s*([}\]])", r"\1", response_str)
            pred_data = json.loads(response_str_fixed)
        except json.JSONDecodeError:
            return {
                "score": 0,
                "reason": "Failed to parse model response as JSON, even after attempting to fix common errors.",
            }

    gt_data = json.loads(answer)
    # extra_info = json.loads(extra_info)

    roll_pred = pred_data.get("roll")
    pitch_pred = pred_data.get("pitch")
    vfov_pred = pred_data.get("vfov")

    if roll_pred is None or pitch_pred is None or vfov_pred is None:
        return {
            "score": 0,
            "reason": "Missing one or more required fields (roll, pitch, vfov) in model response.",
        }

    roll_gt = gt_data["roll"]
    pitch_gt = gt_data["pitch"]
    vfov_gt = gt_data["vfov"]

    roll_unc = extra_info["roll_unc"]
    pitch_unc = extra_info["pitch_unc"]
    vfov_unc = extra_info["vfov_unc"]

    # Score roll with continuous Gaussian-based rating
    roll_diff = abs(roll_pred - roll_gt)
    roll_uncertainty = 1 * roll_unc
    roll_score = np.exp(-(roll_diff**2) / (2 * (roll_uncertainty**2))) if roll_uncertainty > 0 else 0
    roll_score = max(0, min(roll_score, 1))

    # Score pitch with continuous Gaussian-based rating
    pitch_diff = abs(pitch_pred - pitch_gt)
    pitch_uncertainty = 1 * pitch_unc
    pitch_score = np.exp(-(pitch_diff**2) / (2 * (pitch_uncertainty**2))) if pitch_uncertainty > 0 else 0
    pitch_score = max(0, min(pitch_score, 1))

    # Score vfov with continuous Gaussian-based rating
    vfov_diff = abs(vfov_pred - vfov_gt)
    vfov_uncertainty = 1 * vfov_unc
    vfov_score = np.exp(-(vfov_diff**2) / (2 * (vfov_uncertainty**2))) if vfov_uncertainty > 0 else 0
    vfov_score = max(0, min(vfov_score, 1))

    total_score = (roll_score + pitch_score + vfov_score) / 3.0

    result = {
        "score": float(total_score),
        "details": {
            "roll_score": float(roll_score),
            "pitch_score": float(pitch_score),
            "vfov_score": float(vfov_score),
        },
    }
    return result


def multichoiceeval(response, answer, extra_info):
    """
    Calculates the score for the multi-choice metric.
    Args:
        response: The response from the model.
        answer: The answer to the question.
        extra_info: The multi-choice metric must contain the following keys: option.
    Returns:
        A dictionary containing the score and the details.
    """

    def get_option_info(options):
        start_chr = "A"
        all_choices = []
        index2ans = {}
        for i, option in enumerate(options):
            index2ans[chr(ord(start_chr) + i)] = option
            all_choices.append(chr(ord(start_chr) + i))

        return index2ans, all_choices

    def parse_multi_choice_response(response, all_choices, index2ans):
        answer_patterns = [
            r"The correct option is ([A-Z])",
            r"The answer is ([A-Z])",
            r"The correct answer is ([A-Z])",
            r"Option ([A-Z]) is correct",
            r"The right choice is ([A-Z])",
            r"Thus, the answer is ([A-Z])",
            r"Therefore, the correct option is ([A-Z])",
            r"The answer should be ([A-Z])",
            r"Hence, the correct answer is ([A-Z])",
            r"The correct choice is ([A-Z])",
            r"Accordingly, the answer is ([A-Z])",
            r"Consequently, the correct option is ([A-Z])" r"The correct option is ([A-Z])",
            r"The answer is ([A-Z])",
            r"The correct answer is ([A-Z])",
            r"Option ([A-Z]) is correct",
            r"Answer: ([A-Z])",  # 如 "Answer: A"
            r"Answer ([A-Z])",  # 如 "Answer A"
            r"Choice ([A-Z])",  # 如 "Choice A"
            r"Select ([A-Z])",  # 如 "Select A"
            r"([A-Z]) is correct",  # 如 "A is correct"
            r"([A-Z]) is the answer",  # 如 "A is the answer"
            r"([A-Z]) is the correct option",  # 如 "A is the correct option"
            r"Correct: ([A-Z])",  # 如 "Correct: A"
            r"Right: ([A-Z])",  # 如 "Right: A"
            r"<\|begin_of_box\|>([A-Z])<\|end_of_box\|>",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match and match.group(1) in all_choices:
                return match.group(1)

        last_answer_pos = response.rfind("Answer:")
        if last_answer_pos != -1:
            answer_str = response[last_answer_pos + len("Answer:") :].strip()
            matching_options = [option for option in all_choices if option in answer_str]
            if len(matching_options) == 1:
                return matching_options[0]

        last_answer_pos = response.rfind("Answer:")
        if last_answer_pos != -1:
            answer_str = response[last_answer_pos + len("Answer:") :].strip()
            matching_options = [option for option in all_choices if option in answer_str]
            if len(matching_options) == 1:
                return matching_options[0]

        if isinstance(response, str):
            for char in [",", ".", "!", "?", ";", ":", "'"]:
                response = response.strip(char)
            response = " " + response + " "
        else:
            print(response)
            response = ""

        index_ans = True
        ans_with_brack = False
        candidates = []
        for choice in all_choices:
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True

        if len(candidates) == 0:
            for choice in all_choices:
                if f"{choice} " in response:
                    candidates.append(choice)

        if len(candidates) == 0:
            for choice in all_choices:
                if f"{choice}." in response:
                    candidates.append(choice)

        if len(candidates) == 0 and len(response.split()) > 5:
            for index, ans in index2ans.items():
                if ans.lower() in response.lower():
                    candidates.append(index)
                    index_ans = False

        if len(candidates) == 0:
            pred_index = random.choice(all_choices)
        elif len(candidates) > 1:
            start_indexes = []
            if index_ans:
                if ans_with_brack:
                    for can in candidates:
                        index = response.rfind(f"({can})")
                        start_indexes.append(index)
                else:
                    for can in candidates:
                        index = response.rfind(f" {can} ")
                        start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.lower().rfind(index2ans[can].lower())
                    start_indexes.append(index)
            pred_index = candidates[np.argmax(start_indexes)]
        else:
            pred_index = candidates[0]

        return pred_index

    index2ans, all_choices = get_option_info(extra_info["option"])
    parsed_pred = parse_multi_choice_response(response, all_choices, index2ans)
    score = 1 if parsed_pred == answer else 0
    return {"score": score, "details": {"parsed_pred": parsed_pred, "answer": answer}}


def meanrelativeacc(response, answer, extra_info):
    """
    Calculates the score for the numerical metric.
    Args:
        response: The response from the model.
        answer: The answer to the question.
    Returns:
        A dictionary containing the score and the details.
    """

    def abs_dist_norm(pred, target):
        return abs(pred - target) / target

    def mean_relative_accuracy(pred, target, start, end, interval):
        num_pts = (end - start) / interval + 2
        conf_intervs = np.linspace(start, end, int(num_pts))
        accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
        return accuracy.mean()

    def find_number(pred: str):
        pred = pred.replace(",", "")
        coordinate_patterns = [
            r"\(\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\)",  # (x,y)
            r"\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]",  # [x,y]
            r"\{\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\}",  # {x,y}
            r"\b-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\b",  # x,y without brackets
        ]
        modified_pred = pred

        for pattern in coordinate_patterns:
            modified_pred = re.sub(pattern, "", modified_pred)

        match = re.search(r"-?\d+\.?\d*", modified_pred)
        if match:
            return match.group()

        all_matches = list(re.finditer(r"-?\d+\.?\d*", pred))
        if not all_matches:
            return ""

        for i, match in enumerate(all_matches):
            start, end = match.span()
            remaining_text = pred[end:].strip()

            if i == len(all_matches) - 1:
                return match.group()

            if not re.match(r"^,\s*", remaining_text):
                return match.group()

        return all_matches[0].group() if all_matches else ""

    pred_number = find_number(response)
    gt_number = float(answer)
    score = mean_relative_accuracy(float(pred_number), gt_number, 0.5, 0.95, 0.05)
    return {
        "score": score,
        "details": {"pred_number": pred_number, "gt_number": gt_number},
    }


def cogmapeval(response, answer, extra_info):
    """
    Calculates the score for the cogmap metric.
    Args:
        response: The response from the model.
        answer: The answer to the question.
        extra_info: The cogmap metric must contain the following keys: option.
    Returns:
        A dictionary containing the score and the details.
    """
    from lmms_eval.tasks.spatialtreebench.metrics.mindcube_cogmap import (
        mindcube_process_results,
    )

    input_data = {
        "prediction": response,
        "id_type": extra_info["id_type"],
        "ground_truth": answer,
        "cogmap": extra_info["cogmap"],
    }
    result = mindcube_process_results(input_data)
    extra_result = {
        "rotation_invariant_isomorphic": float(result["rotation_invariant_isomorphic"]),
        "overall_similarity": float(result["overall_similarity"]),
        "parsable_json": 1 if result["parsable_json"] else 0,
        "valid_format": 1 if result["valid_format"] else 0,
        "valid_graph": 1 if result["valid_graph"] else 0,
        "coverage": float(result["coverage"]),
        "position_similarity": float(result["position_similarity"]),
        "facing_similarity": float(result["facing_similarity"]),
        "best_rotation": result["best_rotation"]["name"] if result["best_rotation"] else None,
    }
    score = 0.2 * float(result["answer_correct"]) + 0.6 * extra_result["overall_similarity"] + 0.2 * extra_result["rotation_invariant_isomorphic"]
    return {"score": score, "details": extra_result}


def gpteval(response, answer, extra_info):
    """
    Calculates the score for the gpt metric.
    Args:
        response: The response from the model.
        answer: The answer to the question.
        extra_info: The gpt metric must contain the following keys: prompt.
    Returns:
        A dictionary containing the score and the details.
    """
    from lmms_eval.llm_judge import Request, ServerConfig, get_server

    GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "YOUR_MODEL_VERSION")
    API_TYPE = os.getenv("API_TYPE", "azure")
    server_config = ServerConfig(model_name=GPT_EVAL_MODEL_NAME, temperature=0.5, max_tokens=32000)
    server = get_server(server_name=API_TYPE, config=server_config)

    judge_system_prompt = """**[角色]** 
        你是一名严谨的视频运动描述（Video Motion Captioning）阅卷老师。 

        **[任务]** 
        你的任务是严格参考“标准答案 (Groundtruth)”描述，对“模型预测 (Prediction)”的描述进行审核和评分。你必须对描述的**一致性**和**完整性**进行严格评估。 

        **[关键点]** 
        1.  **分析核心要素：** 评分的核心是比较[Prediction]与[Groundtruth]在**运动核心要素**上的一致性。这些要素包括： 
            * **运动主体 (Subject):** 是谁/什么在动？ 
            * **动作 (Action):** 执行了什么动作？（例如：跑、跳、旋转） 
            * **方向/路径 (Direction/Path):** 动作的方向？（例如：从左到右、向上） 
            * **时序 (Temporal Order):** 如果有多个动作，顺序是否正确？（例如：先走，*然后*跳） 
        2.  **提取与评判：** 你需要分析[Prediction]中的上述要素（即“思考过程”），并将其与[Groundtruth]进行对比。评分主要依据“思考过程”的准确性。 
        3.  **严格评分：** 必须严格执行评分细则。任何关键要素的遗漏或错误（尤其是动作和方向）都会导致显著扣分。 
        4.  **格式要求：** 评分依据（reason）必须简短（200字以内），逻辑清晰，并以“综上，学生的答案应得x分”结尾。 
        5. **分数档位：** 参考标准答案评判最终答案，分数值为0分到1分之间的一个两位小数，数值越大分数越高，表明学生回答越正确（最低即为0分，如出现0分仍需扣分的情况，则给0分即可）。 

        **[输出格式]** 
        ( 给出评分并用代码块以“JSON”格式展示，要保留score外部的中括号[[ ]] ) 
        {"answer_score": [[score]], "reason": str} 
        """

    judge_prompt_template = """题目: 
        {prompt} 

        标准答案: 
        {response_reference} 

        学生答案: 
        {response}"""

    def get_eval(user_prompt: str, max_tokens: int, retries: int = 5):
        messages = [
            {
                "role": "system",
                "content": judge_system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]

        custom_config = ServerConfig(model_name=GPT_EVAL_MODEL_NAME, temperature=0.2, max_tokens=max_tokens)

        for attempt in range(retries):
            request = Request(messages=messages, config=custom_config)
            response_obj = server.evaluate(request)
            content = response_obj.content.strip() if response_obj.content else ""
            if content != "":
                return content, response_obj.model_used
        return "", ""

    def parse_score(review):
        try:
            # Extract JSON from the review string
            match = re.search(r"```json\n(.*?)\n```", review, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Fallback for JSON not in a code block
                start = review.find("{")
                end = review.rfind("}")
                if start != -1 and end != -1:
                    json_str = review[start : end + 1]
                else:
                    # No JSON found, raise exception to go to fallback
                    raise ValueError("No JSON found in review")

            data = json.loads(json_str)
            # Robustly get the score from "answer_score"
            score_val = data.get("answer_score")
            if score_val is None:
                raise KeyError("'answer_score' not in JSON")

            # Handle nested lists/values, e.g. [[1]], [1], or 1
            while isinstance(score_val, list):
                if not score_val:
                    raise IndexError("Empty list for 'answer_score'")
                score_val = score_val[0]

            score = float(score_val)
            reason = data.get("reason", "")
            return score, reason
        except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError):
            # Fallback to regex matching for a float
            # This regex will match integers and floats.
            match = re.search(r"(\d+\.\d+|\d+)", review)
            if match:
                try:
                    score = float(match.group(1))
                    return score, "Fallback: Parsed float from text"
                except (ValueError, IndexError):
                    # This should not happen if regex is correct, but for safety
                    pass
            return -1, "JSON parsing failed and no float found in review"

    try:
        if isinstance(extra_info, str):
            extra_info_dict = json.loads(extra_info)
        else:
            extra_info_dict = extra_info
        prompt_text = extra_info_dict.get("prompt", "")
    except (json.JSONDecodeError, AttributeError):
        prompt_text = ""

    user_prompt = judge_prompt_template.format(prompt=prompt_text, response_reference=answer, response=response)

    review, model_used = get_eval(user_prompt, 32000)
    if not review:
        return {
            "score": 0,
            "details": {
                "error": "Failed to get evaluation from judge model.",
                "model_used": model_used,
            },
        }

    score, reason = parse_score(review)

    if score == -1:
        return {
            "score": 0,
            "details": {
                "error": f"Failed to parse score from review: {reason}",
                "model_used": model_used,
            },
        }

    return {"score": score, "details": {"reason": reason, "model_used": model_used}}


def judgemodel(response, answer, extra_info):
    """
    Calculates the score for the judge model.
    Args:
        response: The response from the model.
        answer: The answer to the question.
        extra_info: The judge model must contain the following keys: prompt.
    Returns:
        A dictionary containing the score and the details.
    """

    def parse_last_yes_no(text: str) -> str:
        if not text:
            return "other"

        lower_text = text.lower()

        last_yes_index = lower_text.rfind("yes")
        last_no_index = lower_text.rfind("no")

        if last_yes_index == -1 and last_no_index == -1:
            return "other"
        if last_yes_index > last_no_index:
            return "yes"
        else:
            return "no"

    gt_list = answer.lower()
    predict = parse_last_yes_no(response)
    score = 1 if gt_list == predict else 0
    return {"score": score, "details": {"predict": predict, "gt": gt_list}}


def aff_mask_metric(response: str, answer: str, extra_info: str = None):
    """
    计算aff_mask_metric
    """
    import base64
    import re
    from typing import Optional, Tuple

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return 0.0

    def decode_mask(mask_b64: str):
        if not mask_b64:
            return None
        b64_str = mask_b64
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,", 1)[-1]
        try:
            mask_bytes = base64.b64decode(b64_str, validate=False)
        except Exception:
            try:
                mask_bytes = base64.b64decode(b64_str + "===")
            except Exception:
                return None
        nparr = np.frombuffer(mask_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img

    def extract_last_point(text: str) -> Optional[Tuple[float, float]]:
        if not isinstance(text, str):
            return None
        s = text.strip()
        candidates: Tuple[int, float, float] | None = None

        # Patterns to capture (x, y) in various forms; choose the one closest to the end
        patterns = [
            (
                re.compile(r"(?i)x\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\D+?y\s*[:=]\s*([-+]?\d+(?:\.\d+)?)"),
                "xy",
            ),
            (
                re.compile(r"(?i)y\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\D+?x\s*[:=]\s*([-+]?\d+(?:\.\d+)?)"),
                "yx",
            ),
            (
                re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)"),
                "xy",
            ),
            (
                re.compile(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]"),
                "xy",
            ),
            (re.compile(r"([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)"), "xy"),
        ]

        for regex, kind in patterns:
            for m in regex.finditer(s):
                try:
                    a = float(m.group(1))
                    b = float(m.group(2))
                    x, y = (a, b) if kind == "xy" else (b, a)
                except Exception:
                    continue
                end_pos = m.end()
                if candidates is None or end_pos > candidates[0]:
                    candidates = (end_pos, x, y)

        if candidates is None:
            return None
        return (candidates[1], candidates[2])

    # 1) Parse last coordinate pair from predict (string)
    point = extract_last_point(response)
    if point is None:
        return 0.0
    x_pred, y_pred = point

    # 2) Decode single base64 mask from gt_list (string)
    mask = decode_mask(answer)
    if mask is None or mask.size == 0:
        return 0.0

    h, w = mask.shape[:2]
    # Heuristic: treat as normalized if within [0,1]
    if 0.0 <= x_pred <= 1.0 and 0.0 <= y_pred <= 1.0:
        col = int(round(x_pred * (w - 1)))
        row = int(round(y_pred * (h - 1)))
    else:
        col = int(round(x_pred))
        row = int(round(y_pred))

    col = max(0, min(w - 1, col))
    row = max(0, min(h - 1, row))

    pixel_val = float(mask[row, col])
    score = pixel_val / 255.0
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return {
        "score": float(score),
        "details": {
            "x_pred": x_pred,
            "y_pred": y_pred,
            "col": col,
            "row": row,
            "pixel_val": pixel_val,
        },
    }


def actions2cam_response(instructions, initial_pose, extra_info):
    """
    Converts a model's response with symbolic actions into a sequence of camera poses.
    This function is based on the logic from `utils.cam2actions.actions2cam`.
    """
    SYMBOL_TO_ACTION = {
        "W": "Dolly In",
        "S": "Dolly Out",
        "A": "Truck Left",
        "D": "Truck Right",
        "space": "Pedestal Up",
        "shift": "Pedestal Down",
        "←": "Pan Left",
        "→": "Pan Right",
        "↑": "Tilt Up",
        "↓": "Tilt Down",
        "↺": "Roll CCW",
        "↻": "Roll CW",
        "STOP": "Stay",
    }

    predicted_poses_list = [initial_pose.copy()]
    current_pose = initial_pose.copy()

    action_keys = sorted([k for k in extra_info if "->" in k], key=lambda x: int(x.split("->")[0]))

    # Handle cases where response keys might not be perfectly ordered or named
    try:
        predicted_keys = sorted(instructions.keys(), key=lambda x: int(re.search(r"\d+", x).group()))
    except (AttributeError, ValueError):
        # Fallback to alphabetical sort if no numbers are found in keys
        predicted_keys = sorted(instructions.keys())

    for i, pred_key in enumerate(predicted_keys):
        if i >= len(action_keys):
            print(f"Warning: Model produced more steps ({len(predicted_keys)}) than ground truth ({len(action_keys)}). Truncating.")
            break

        gt_key = action_keys[i]
        segment_info = extra_info[gt_key]
        translation_step_size = segment_info.get("translation_step_size", 0.02)
        rotation_step_size_rad = segment_info.get("rotation_step_size", np.deg2rad(0.5))

        segment_data = instructions[pred_key]
        if not isinstance(segment_data, dict):
            print(f"Warning: Segment data for '{pred_key}' is not a dictionary. Skipping.")
            continue

        actions = segment_data.get("actions", [])
        step_nums = segment_data.get("step_nums", [])

        current_pos = current_pose[:3]
        current_rot = R.from_quat(current_pose[3:])

        dx, dy, dz = 0.0, 0.0, 0.0
        yaw_change, pitch_change, roll_change = 0.0, 0.0, 0.0

        for j, action_symbol in enumerate(actions):
            action_name = SYMBOL_TO_ACTION.get(action_symbol)
            if not action_name:
                print(f"Warning: Unknown action symbol '{action_symbol}' in step {pred_key}. Skipping.")
                continue

            action_step_nums = step_nums[j] if j < len(step_nums) else 1

            if action_name == "Dolly In":
                dz += translation_step_size * action_step_nums
            elif action_name == "Dolly Out":
                dz -= translation_step_size * action_step_nums
            elif action_name == "Truck Right":
                dx += translation_step_size * action_step_nums
            elif action_name == "Truck Left":
                dx -= translation_step_size * action_step_nums
            elif action_name == "Pedestal Up":
                dy -= translation_step_size * action_step_nums
            elif action_name == "Pedestal Down":
                dy += translation_step_size * action_step_nums
            elif action_name == "Pan Right":
                yaw_change += rotation_step_size_rad * action_step_nums
            elif action_name == "Pan Left":
                yaw_change -= rotation_step_size_rad * action_step_nums
            elif action_name == "Tilt Up":
                pitch_change += rotation_step_size_rad * action_step_nums
            elif action_name == "Tilt Down":
                pitch_change -= rotation_step_size_rad * action_step_nums
            elif action_name == "Roll CW":
                roll_change += rotation_step_size_rad * action_step_nums
            elif action_name == "Roll CCW":
                roll_change -= rotation_step_size_rad * action_step_nums

        # Apply translation in camera's local frame, then transform to world frame
        local_delta_pos = np.array([dx, dy, dz])
        new_pos = current_pos + local_delta_pos

        # Apply rotation
        delta_rot = R.from_euler("zyx", [yaw_change, pitch_change, roll_change])
        new_rot = delta_rot * current_rot

        new_pose = np.zeros(7)
        new_pose[:3] = new_pos
        quat = new_rot.as_quat()
        quat = quat / (np.linalg.norm(quat) + 1e-12)
        new_pose[3:] = quat

        predicted_poses_list.append(new_pose)
        current_pose = new_pose

    return np.array(predicted_poses_list)


def manipulateeval(response, answer, extra_info):
    try:
        predicted_instructions = None
        json_string = ""

        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_string = match.group(1)
            else:
                start_index = response.find("{")
                end_index = response.rfind("}")
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_string = response[start_index : end_index + 1]
                else:
                    json_string = response

        try:
            predicted_instructions = json.loads(json_string)
        except json.JSONDecodeError:
            try:
                fixed_json_string = re.sub(r",\s*([}\]])", r"\1", json_string)
                predicted_instructions = json.loads(fixed_json_string)
            except json.JSONDecodeError:
                try:
                    predicted_instructions = ast.literal_eval(json_string)
                except (ValueError, SyntaxError):
                    pass  # Parsing failed

        if not isinstance(predicted_instructions, dict):
            raise ValueError("Parsed object is not a dictionary or could not be found.")

    except (ValueError, SyntaxError):
        print(f"Error: Could not parse response into a dictionary: {response}")
        return {"score": 0, "error": "Invalid response format"}

    optimal_poses = np.array(extra_info["optimal_rel_pose"])
    initial_pose = optimal_poses[0]
    predicted_poses = actions2cam_response(predicted_instructions, initial_pose, extra_info)

    if len(predicted_poses) == 0:
        return {"score": 0, "error": "Predicted trajectory is empty"}

    action_keys = sorted([k for k in extra_info if "->" in k], key=lambda x: int(x.split("->")[0]))
    num_predicted_segments = len(predicted_poses) - 1
    gt_poses_for_comparison = optimal_poses[: num_predicted_segments + 1]  # Ground truth poses to compare against
    predicted_final_pose = predicted_poses[-1]
    gt_final_pose = gt_poses_for_comparison[-1]
    num_total_segments = len(action_keys)

    # Calculate trajectory length (how much movement was required)
    trajectory_length = np.linalg.norm(gt_final_pose[:3] - initial_pose[:3])
    required_movement = trajectory_length > 0.1  # If more than 10cm movement required
    required_rotation = False
    if len(optimal_poses) > 1:
        # Check if rotation was required during the trajectory
        for i in range(1, min(5, len(optimal_poses))):  # Check first few steps
            prev_rot = R.from_quat(optimal_poses[i - 1][3:])
            curr_rot = R.from_quat(optimal_poses[i][3:])
            rot_diff = (prev_rot.inv() * curr_rot).magnitude()
            if rot_diff > np.deg2rad(5):  # If more than 5 degrees rotation between steps
                required_rotation = True
                break

    # --- Component 1: Trajectory Direction Score (Weight: 0.25) ---
    # This measures if the overall direction of movement is correct.
    gt_traj_vec = gt_final_pose[:3] - initial_pose[:3]
    pred_traj_vec = predicted_final_pose[:3] - initial_pose[:3]

    norm_gt = np.linalg.norm(gt_traj_vec)
    norm_pred = np.linalg.norm(pred_traj_vec)

    direction_score = 0.0
    if norm_gt < 1e-6 and norm_pred < 1e-6:
        direction_score = 1.0  # Both stayed put, perfect direction match.
    elif norm_gt > 1e-6 and norm_pred > 1e-6:
        # Use cosine similarity for direction comparison
        cosine_sim = np.dot(gt_traj_vec, pred_traj_vec) / (norm_gt * norm_pred)
        direction_score = max(0, cosine_sim)  # Score is 0 if directions are > 90 degrees apart.
    # If one moved and the other didn't, the score remains 0.

    # --- Component 2: Final Position Score (Weight: 0.25) ---
    # This measures the distance from the predicted endpoint to the ground truth endpoint.
    final_position_error = np.linalg.norm(predicted_final_pose[:3] - gt_final_pose[:3])
    # Score is based on an exponential decay function.
    # It's scaled so a 0.2m error (old threshold) gives a score of ~0.5.
    position_score = np.exp(-13.863 * final_position_error)

    # --- Component 3: Final Rotation Score (Weight: 0.25) ---
    # This measures the difference in final camera orientation.
    pred_final_rot = R.from_quat(predicted_final_pose[3:])
    gt_final_rot = R.from_quat(gt_final_pose[3:])
    final_rotation_error_rad = (pred_final_rot.inv() * gt_final_rot).magnitude()
    final_rotation_error_deg = np.rad2deg(final_rotation_error_rad)
    rotation_score = np.exp(-0.0154 * (final_rotation_error_deg / 5.0) ** 2)

    # Apply cumulative penalties for both movement and rotation tasks
    relevance_penalty = 1.0

    # --- Component 1: Movement Penalty ---
    gt_movement_magnitude = np.linalg.norm(gt_final_pose[:3] - initial_pose[:3])  # Calculate ground truth movement magnitude
    pred_movement_magnitude = np.linalg.norm(predicted_final_pose[:3] - initial_pose[:3])  # Calculate predicted movement magnitude

    # If ground truth required significant movement (>10cm) but model moved little (<5cm)
    if gt_movement_magnitude > 0.1 and pred_movement_magnitude < 0.03:
        relevance_penalty *= 0.7  # Significant penalty for not moving when needed

    # --- Component 2: Rotation Penalty ---
    # Calculate ground truth rotation magnitude
    gt_rotation_required = False
    if len(optimal_poses) > 1:
        total_gt_rotation = 0
        for i in range(1, len(optimal_poses)):
            prev_rot = R.from_quat(optimal_poses[i - 1][3:])
            curr_rot = R.from_quat(optimal_poses[i][3:])
            rot_diff = (prev_rot.inv() * curr_rot).magnitude()
            total_gt_rotation += rot_diff

        # Consider significant rotation as more than 5 degrees in total
        if total_gt_rotation > np.deg2rad(5):
            gt_rotation_required = True

    # Calculate predicted rotation magnitude
    model_rotated = False
    if len(predicted_poses) > 1:
        total_pred_rotation = 0
        for i in range(1, len(predicted_poses)):
            prev_rot = R.from_quat(predicted_poses[i - 1][3:])
            curr_rot = R.from_quat(predicted_poses[i][3:])
            rot_diff = (prev_rot.inv() * curr_rot).magnitude()
            total_pred_rotation += rot_diff

        # Consider model rotated if it rotated more than 2 degrees in total
        if total_pred_rotation > np.deg2rad(2):
            model_rotated = True

    # If ground truth required significant rotation but model didn't rotate enough
    if gt_rotation_required and not model_rotated:
        relevance_penalty *= 0.5  # Moderate penalty for not rotating when needed

    # --- Component 4: Distance Ratio Score (Weight: 0.1) ---
    # This measures how much of the required movement the model actually performed
    distance_ratio_score = 0.0
    if gt_movement_magnitude > 1e-6:  # Avoid division by zero
        # Calculate the ratio of predicted movement to required movement
        movement_ratio = pred_movement_magnitude / gt_movement_magnitude

        # Map the ratio to a score using an exponential function that peaks at ratio = 1.0
        # This rewards models that move approximately the right distance
        distance_ratio_score = np.exp(-15.0 * (movement_ratio - 1.0) ** 2)
    else:
        # If no movement was required, give full score if model stayed put
        if pred_movement_magnitude < 1e-6:
            distance_ratio_score = 1.0

    # --- Final Combined Score ---
    # Combine all components and apply relevance penalty
    # Adjusted weights to include distance ratio score
    final_score = (0.4 * direction_score + 0.35 * position_score + 0.25 * rotation_score) * relevance_penalty

    # Ensure score is within valid range
    final_score = max(0.0, min(1.0, final_score))

    # For binary success, we still check against the original strict thresholds
    POS_SUCCESS_THRESHOLD = 0.2  # meters
    ROT_SUCCESS_THRESHOLD = 10  # degrees
    is_successful_binary = (num_predicted_segments == num_total_segments) and (final_position_error < POS_SUCCESS_THRESHOLD) and (final_rotation_error_deg < ROT_SUCCESS_THRESHOLD)

    result = {
        "score": float(final_score),
        "scores": {
            "direction_score": float(direction_score),
            "position_score": float(position_score),
            "rotation_score": float(rotation_score),
            "distance_ratio_score": float(distance_ratio_score),
        },
        "is_successful_binary": bool(is_successful_binary),
        "final_position_error_m": float(final_position_error),
        "final_rotation_error_deg": float(final_rotation_error_deg),
        "predicted_segments": int(num_predicted_segments),
        "total_segments": int(num_total_segments),
        "gt_movement_magnitude": float(gt_movement_magnitude),
        "pred_movement_magnitude": float(pred_movement_magnitude),
    }
    return result


def action2cam_response(instructions, initial_pose, extra_info):
    SYMBOL_TO_ACTION = {
        "W": "Dolly In",
        "S": "Dolly Out",
        "A": "Truck Left",
        "D": "Truck Right",
        "space": "Pedestal Up",
        "shift": "Pedestal Down",
        "←": "Pan Left",
        "→": "Pan Right",
        "↑": "Tilt Up",
        "↓": "Tilt Down",
        "↺": "Roll CCW",
        "↻": "Roll CW",
        "STOP": "Stay",
    }
    predicted_poses_list = [initial_pose.copy()]
    current_pose = initial_pose.copy()

    action_keys = sorted([k for k in extra_info if "->" in k], key=lambda x: int(x.split("->")[0]))

    gt_key = action_keys[-1]
    segment_info = extra_info[gt_key]
    translation_step_size = segment_info.get("translation_step_size", 0.02)
    rotation_step_size_rad = segment_info.get("rotation_step_size", np.deg2rad(0.5))

    segment_data = instructions
    actions = segment_data.get("actions", [])
    step_nums = segment_data.get("step_nums", [])

    current_pos = current_pose[:3]
    current_rot = R.from_quat(current_pose[3:])

    dx, dy, dz = 0.0, 0.0, 0.0
    yaw_change, pitch_change, roll_change = 0.0, 0.0, 0.0

    for j, action_symbol in enumerate(actions):
        action_name = SYMBOL_TO_ACTION.get(action_symbol)
        if not action_name:
            print(f"Warning: Unknown action symbol '{action_symbol}'. Skipping.")
            continue

        action_step_nums = step_nums[j] if j < len(step_nums) else 1

        if action_name == "Dolly In":
            dz += translation_step_size * action_step_nums
        elif action_name == "Dolly Out":
            dz -= translation_step_size * action_step_nums
        elif action_name == "Truck Right":
            dx += translation_step_size * action_step_nums
        elif action_name == "Truck Left":
            dx -= translation_step_size * action_step_nums
        elif action_name == "Pedestal Up":
            dy -= translation_step_size * action_step_nums
        elif action_name == "Pedestal Down":
            dy += translation_step_size * action_step_nums
        elif action_name == "Pan Right":
            yaw_change += rotation_step_size_rad * action_step_nums
        elif action_name == "Pan Left":
            yaw_change -= rotation_step_size_rad * action_step_nums
        elif action_name == "Tilt Up":
            pitch_change += rotation_step_size_rad * action_step_nums
        elif action_name == "Tilt Down":
            pitch_change -= rotation_step_size_rad * action_step_nums
        elif action_name == "Roll CW":
            roll_change += rotation_step_size_rad * action_step_nums
        elif action_name == "Roll CCW":
            roll_change -= rotation_step_size_rad * action_step_nums

    # Apply translation in camera's local frame, then transform to world frame
    local_delta_pos = np.array([dx, dy, dz])
    new_pos = current_pos + local_delta_pos

    # Apply rotation
    delta_rot = R.from_euler("zyx", [yaw_change, pitch_change, roll_change])
    new_rot = delta_rot * current_rot

    new_pose = np.zeros(7)
    new_pose[:3] = new_pos
    quat = new_rot.as_quat()
    quat = quat / (np.linalg.norm(quat) + 1e-12)
    new_pose[3:] = quat
    current_pose = new_pose
    return np.array(current_pose)


def agenticnaveval(response, answer, extra_info):
    try:
        predicted_instructions = None
        json_string = ""

        # Strategy 1: Look for a JSON markdown block
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            # Strategy 2: Look for a generic markdown block
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_string = match.group(1)
            else:
                # Strategy 3: Look for content between the first '{' and last '}'
                start_index = response.find("{")
                end_index = response.rfind("}")
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_string = response[start_index : end_index + 1]
                else:
                    # Strategy 4: Use the whole response as a last resort
                    json_string = response

        # Attempt to parse the extracted string
        try:
            predicted_instructions = json.loads(json_string)
        except json.JSONDecodeError:
            # Try to fix common JSON errors, like trailing commas
            try:
                fixed_json_string = re.sub(r",\s*([}\]])", r"\1", json_string)
                predicted_instructions = json.loads(fixed_json_string)
            except json.JSONDecodeError:
                # If JSON parsing fails, try ast.literal_eval for Python dicts
                try:
                    predicted_instructions = ast.literal_eval(json_string)
                except (ValueError, SyntaxError):
                    pass  # Parsing failed

        if not isinstance(predicted_instructions, dict):
            raise ValueError("Parsed object is not a dictionary or could not be found.")

    except (ValueError, SyntaxError):
        print(f"Error: Could not parse response into a dictionary: {response}")
        return {"score": 0, "error": "Invalid response format"}

    optimal_poses = np.array(extra_info["optimal_rel_pose"])
    initial_pose = optimal_poses[-2]

    # Reconstruct the full predicted trajectory using the model's response
    predicted_poses = action2cam_response(predicted_instructions, initial_pose, extra_info)

    if len(predicted_poses) == 0:
        return {"score": 0, "error": "Predicted trajectory is empty"}

    action_keys = sorted([k for k in extra_info if "->" in k], key=lambda x: int(x.split("->")[0]))
    num_predicted_segments = len(predicted_poses) - 1

    gt_poses_for_comparison = optimal_poses[: num_predicted_segments + 1]  # Ground truth poses to compare against

    predicted_final_pose = predicted_poses
    gt_final_pose = gt_poses_for_comparison[-1]

    trajectory_length = np.linalg.norm(gt_final_pose[:3] - initial_pose[:3])
    required_movement = trajectory_length > 0.1  # If more than 10cm movement required
    required_rotation = False
    if len(optimal_poses) > 1:
        for i in range(1, min(5, len(optimal_poses))):  # Check first few steps
            prev_rot = R.from_quat(optimal_poses[i - 1][3:])
            curr_rot = R.from_quat(optimal_poses[i][3:])
            rot_diff = (prev_rot.inv() * curr_rot).magnitude()
            if rot_diff > np.deg2rad(5):  # If more than 5 degrees rotation between steps
                required_rotation = True
                break

    gt_traj_vec = gt_final_pose[:3] - initial_pose[:3]
    pred_traj_vec = predicted_final_pose[:3] - initial_pose[:3]

    norm_gt = np.linalg.norm(gt_traj_vec)
    norm_pred = np.linalg.norm(pred_traj_vec)

    direction_score = 0.0
    if norm_gt < 1e-6 and norm_pred < 1e-6:
        direction_score = 1.0  # Both stayed put, perfect direction match.
    elif norm_gt > 1e-6 and norm_pred > 1e-6:
        # Use cosine similarity for direction comparison
        cosine_sim = np.dot(gt_traj_vec, pred_traj_vec) / (norm_gt * norm_pred)
        direction_score = max(0, cosine_sim)  # Score is 0 if directions are > 90 degrees apart.

    final_position_error = np.linalg.norm(predicted_final_pose[:3] - gt_final_pose[:3])
    position_score = np.exp(-13.863 * final_position_error)

    pred_final_rot = R.from_quat(predicted_final_pose[3:])
    gt_final_rot = R.from_quat(gt_final_pose[3:])
    final_rotation_error_rad = (pred_final_rot.inv() * gt_final_rot).magnitude()
    final_rotation_error_deg = np.rad2deg(final_rotation_error_rad)
    rotation_score = np.exp(-0.0154 * (final_rotation_error_deg / 5.0) ** 2)

    # --- Final Combined Score ---
    # Combine all components and apply relevance penalty
    # Adjusted weights to include distance ratio score
    final_score = 0.4 * direction_score + 0.35 * position_score + 0.25 * rotation_score

    # Ensure score is within valid range
    final_score = max(0.0, min(1.0, final_score))

    result = {
        "score": float(final_score),
        "scores": {
            "direction_score": float(direction_score),
            "position_score": float(position_score),
            "rotation_score": float(rotation_score),
        },
        "final_position_error_m": float(final_position_error),
        "final_rotation_error_deg": float(final_rotation_error_deg),
    }
    return result
