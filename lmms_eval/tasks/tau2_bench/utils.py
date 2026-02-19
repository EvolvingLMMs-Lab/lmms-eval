import ast
import copy
import json
import re

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
SUBMIT_PATTERN = re.compile(r"<submit>(.*?)</submit>", re.DOTALL)


def tau2_doc_to_visual(doc):
    return []


def tau2_doc_to_target(doc):
    return doc.get("target_state", {})


def _extract_tag_payload(pattern, text):
    matches = pattern.findall(text or "")
    payloads = []
    for match in matches:
        candidate = match.strip()
        if not candidate:
            continue
        try:
            payloads.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return payloads


def _parse_json_like(candidate):
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, SyntaxError):
            return None
    return None


def _extract_braced_objects(text):
    if not text:
        return []

    objects = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start : idx + 1])
                start = -1

    return objects


def _extract_submit_payloads(text):
    payloads = _extract_tag_payload(SUBMIT_PATTERN, text)
    if payloads:
        return payloads

    for obj_text in _extract_braced_objects(text):
        obj = _parse_json_like(obj_text)
        if not isinstance(obj, dict):
            continue
        if "final" in obj:
            payloads.append(obj)
        elif "submit" in obj and isinstance(obj["submit"], dict):
            payloads.append(obj["submit"])

    return payloads


def _extract_tool_payloads(text):
    payloads = _extract_tag_payload(TOOL_CALL_PATTERN, text)
    if payloads:
        return payloads

    for obj_text in _extract_braced_objects(text):
        obj = _parse_json_like(obj_text)
        if not isinstance(obj, dict):
            continue

        if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
            for call in obj["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                if "function" in call and isinstance(call["function"], dict):
                    fn = call["function"]
                    name = fn.get("name")
                    arguments = fn.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = _parse_json_like(arguments) or {}
                    if name:
                        payloads.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})

        name = obj.get("name") or obj.get("tool_name")
        if name and "final" not in obj:
            arguments = obj.get("arguments", obj.get("args", {}))
            if isinstance(arguments, str):
                arguments = _parse_json_like(arguments) or {}
            payloads.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})

    return payloads


def _apply_tool(tool_name, arguments, state):
    if tool_name == "get_line_status":
        return {
            "abroad": state["abroad"],
            "airplane_mode": state["airplane_mode"],
            "roaming_enabled": state["roaming_enabled"],
            "mobile_data_working": state["mobile_data_working"],
        }
    if tool_name == "disable_airplane_mode":
        state["airplane_mode"] = False
        if state["roaming_enabled"] and state["abroad"]:
            state["mobile_data_working"] = True
        return {"ok": True}
    if tool_name == "enable_roaming":
        state["roaming_enabled"] = True
        if (not state["airplane_mode"]) and state["abroad"]:
            state["mobile_data_working"] = True
        return {"ok": True}
    if tool_name == "reset_network":
        state["mobile_data_working"] = state["roaming_enabled"] and (not state["airplane_mode"]) and state["abroad"]
        return {"ok": True}
    return {"error": f"unknown tool '{tool_name}'"}


def _build_agent_prompt(doc, state, tool_result=None):
    tool_specs = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in doc["tools"]])
    tool_result_text = ""
    if tool_result is not None:
        tool_result_text = f"\nTool result: {json.dumps(tool_result, ensure_ascii=False)}"

    target_state = doc.get("target_state", {})
    goal_reached = isinstance(target_state, dict) and target_state and all(state.get(key) == value for key, value in target_state.items())

    if goal_reached:
        return (
            f"{doc['user_query']}\n"
            f"Current state: {json.dumps(state, ensure_ascii=False)}{tool_result_text}\n"
            "Target state is already satisfied. Do not call any more tools.\n"
            "Output exactly one submit in this format and nothing else:\n"
            '<submit>{"final":"mobile data works now"}</submit>'
        )

    return (
        f"{doc['user_query']}\n"
        f"Current state: {json.dumps(state, ensure_ascii=False)}{tool_result_text}\n"
        "You are in an agentic telecom support loop.\n"
        "Either call exactly one tool or submit final answer.\n"
        'Tool format: <tool_call>{"name":"tool_name","arguments":{}}</tool_call>\n'
        'Submit format: <submit>{"final":"..."}</submit>\n'
        f"Available tools:\n{tool_specs}"
    )


def tau2_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
    if round_idx is None:
        init_state = copy.deepcopy(doc["initial_state"])
        return _build_agent_prompt(doc, init_state)

    state_info = previous_round_info or {
        "state": copy.deepcopy(doc["initial_state"]),
        "tool_calls": 0,
        "valid_tool_calls": 0,
        "invalid_steps": 0,
        "last_tool_result": None,
    }
    state = state_info["state"]
    model_response = previous_output[-1] if previous_output else ""

    tool_payloads = _extract_tool_payloads(model_response)
    if tool_payloads:
        for tool_call in tool_payloads:
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = _parse_json_like(arguments) or {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_result = _apply_tool(tool_name, arguments, state)
            state_info["tool_calls"] += 1
            if isinstance(tool_result, dict) and "error" in tool_result:
                state_info["invalid_steps"] = state_info.get("invalid_steps", 0) + 1
            else:
                state_info["valid_tool_calls"] = state_info.get("valid_tool_calls", 0) + 1
            state_info["last_tool_result"] = tool_result

    submit_payloads = _extract_submit_payloads(model_response)
    if submit_payloads:
        expected = doc["target_state"]
        success = all(state.get(key) == value for key, value in expected.items())
        final_payload = {
            "success": success,
            "tool_calls": state_info["tool_calls"],
            "valid_tool_calls": state_info.get("valid_tool_calls", 0),
            "invalid_steps": state_info.get("invalid_steps", 0),
            "state": state,
            "submit": submit_payloads[-1],
            "trace": previous_output,
        }
        return None, None, True, [json.dumps(final_payload, ensure_ascii=False)], state_info

    if tool_payloads:
        next_prompt = _build_agent_prompt(doc, state, tool_result=state_info.get("last_tool_result"))
        return [], next_prompt, False, previous_output, state_info

    no_action_result = {
        "error": "no valid <tool_call> or <submit> found",
        "hint": "emit exactly one tool call or submit",
    }
    state_info["invalid_steps"] = state_info.get("invalid_steps", 0) + 1
    next_prompt = _build_agent_prompt(doc, state, tool_result=no_action_result)
    state_info["last_tool_result"] = no_action_result
    return [], next_prompt, False, previous_output, state_info


def tau2_process_results(doc, results):
    raw = results[0] if results else ""
    success = 0.0
    tool_calls = 0.0
    trace_step_validity = 0.0
    trace_state_progress = 0.0
    trace_termination_quality = 0.0
    trace_quality = 0.0

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            success = 1.0 if payload.get("success") else 0.0
            tool_calls = float(payload.get("tool_calls", 0))

            valid_calls = float(payload.get("valid_tool_calls", tool_calls))
            invalid_steps = float(payload.get("invalid_steps", 0.0))
            denom = max(valid_calls + invalid_steps, 1.0)
            trace_step_validity = valid_calls / denom

            state = payload.get("state", {})
            target = doc.get("target_state", {})
            if isinstance(state, dict) and isinstance(target, dict) and target:
                matched = sum(1 for key, expected in target.items() if state.get(key) == expected)
                trace_state_progress = matched / len(target)

            trace_termination_quality = 1.0 if payload.get("submit") else 0.0
            trace_quality = (trace_step_validity + trace_state_progress + trace_termination_quality) / 3.0
        except json.JSONDecodeError:
            success = 0.0
            tool_calls = 0.0

    return {
        "tau2_success": success,
        "tau2_avg_tool_calls": tool_calls,
        "tau2_trace_quality": trace_quality,
        "tau2_trace_step_validity": trace_step_validity,
        "tau2_trace_state_progress": trace_state_progress,
        "tau2_trace_termination_quality": trace_termination_quality,
    }


def tau2_aggregate_success(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def tau2_aggregate_avg_tool_calls(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def tau2_aggregate_trace_quality(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def tau2_aggregate_trace_step_validity(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def tau2_aggregate_trace_state_progress(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def tau2_aggregate_trace_termination_quality(results):
    if not results:
        return 0.0
    return sum(results) / len(results)
