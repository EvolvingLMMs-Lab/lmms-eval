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

    state_info = previous_round_info or {"state": copy.deepcopy(doc["initial_state"]), "tool_calls": 0, "last_tool_result": None}
    state = state_info["state"]
    model_response = previous_output[-1] if previous_output else ""

    submit_payloads = _extract_tag_payload(SUBMIT_PATTERN, model_response)
    if submit_payloads:
        expected = doc["target_state"]
        success = all(state.get(key) == value for key, value in expected.items())
        final_payload = {
            "success": success,
            "tool_calls": state_info["tool_calls"],
            "state": state,
            "submit": submit_payloads[-1],
        }
        return None, None, True, [json.dumps(final_payload, ensure_ascii=False)], None

    tool_payloads = _extract_tag_payload(TOOL_CALL_PATTERN, model_response)
    if tool_payloads:
        tool_call = tool_payloads[-1]
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        tool_result = _apply_tool(tool_name, arguments, state)
        state_info["tool_calls"] += 1
        state_info["last_tool_result"] = tool_result
        next_prompt = _build_agent_prompt(doc, state, tool_result=tool_result)
        return [], next_prompt, False, previous_output, state_info

    no_action_result = {
        "error": "no valid <tool_call> or <submit> found",
        "hint": "emit exactly one tool call or submit",
    }
    next_prompt = _build_agent_prompt(doc, state, tool_result=no_action_result)
    state_info["last_tool_result"] = no_action_result
    return [], next_prompt, False, previous_output, state_info


def tau2_process_results(doc, results):
    raw = results[0] if results else ""
    success = 0.0
    tool_calls = 0.0

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            success = 1.0 if payload.get("success") else 0.0
            tool_calls = float(payload.get("tool_calls", 0))
        except json.JSONDecodeError:
            success = 0.0
            tool_calls = 0.0

    return {
        "tau2_success": success,
        "tau2_avg_tool_calls": tool_calls,
    }


def tau2_aggregate_success(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def tau2_aggregate_avg_tool_calls(results):
    if not results:
        return 0.0
    return sum(results) / len(results)
