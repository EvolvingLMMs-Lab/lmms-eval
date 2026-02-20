import copy
import json
import re

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
SUBMIT_PATTERN = re.compile(r"<submit>(.*?)</submit>", re.DOTALL)


def vending_doc_to_visual(doc):
    return []


def vending_doc_to_target(doc):
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
    if tool_name == "get_vending_status":
        return {
            "cash": state["cash"],
            "inventory": state["inventory"],
            "price": state["price"],
            "days_elapsed": state["days_elapsed"],
        }

    if tool_name == "set_price":
        new_price = float(arguments.get("price", state["price"]))
        if new_price <= 0:
            return {"error": "price must be positive"}
        state["price"] = new_price
        return {"ok": True, "price": state["price"]}

    if tool_name == "restock":
        units = int(arguments.get("units", 0))
        if units <= 0:
            return {"error": "units must be > 0"}
        total_cost = units * state["cost_per_unit"]
        if total_cost > state["cash"]:
            return {"error": "insufficient cash", "needed": total_cost, "cash": state["cash"]}
        state["cash"] -= total_cost
        state["inventory"] += units
        return {"ok": True, "inventory": state["inventory"], "cash": state["cash"]}

    if tool_name == "simulate_days":
        days = int(arguments.get("days", 1))
        if days <= 0:
            return {"error": "days must be > 0"}

        total_demand = int(state["daily_demand"] * days)
        sold_units = min(total_demand, state["inventory"])
        revenue = sold_units * state["price"]

        state["inventory"] -= sold_units
        state["cash"] += revenue
        state["days_elapsed"] += days

        return {
            "ok": True,
            "sold_units": sold_units,
            "revenue": revenue,
            "cash": state["cash"],
            "inventory": state["inventory"],
            "days_elapsed": state["days_elapsed"],
        }

    return {"error": f"unknown tool '{tool_name}'"}


def _build_agent_prompt(doc, state, tool_result=None):
    tool_specs = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in doc["tools"]])
    tool_result_text = ""
    if tool_result is not None:
        tool_result_text = f"\nTool result: {json.dumps(tool_result, ensure_ascii=False)}"

    return (
        f"{doc['user_query']}\n"
        f"Current simulator state: {json.dumps(state, ensure_ascii=False)}{tool_result_text}\n"
        "You are in an agentic vending operation loop.\n"
        "Either call exactly one tool or submit final answer.\n"
        'Tool format: <tool_call>{"name":"tool_name","arguments":{}}</tool_call>\n'
        'Submit format: <submit>{"final":"..."}</submit>\n'
        f"Available tools:\n{tool_specs}"
    )


def vending_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
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

    tool_payloads = _extract_tag_payload(TOOL_CALL_PATTERN, model_response)
    if tool_payloads:
        for tool_call in tool_payloads:
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
            tool_result = _apply_tool(tool_name, arguments, state)
            state_info["tool_calls"] += 1
            if isinstance(tool_result, dict) and "error" in tool_result:
                state_info["invalid_steps"] = state_info.get("invalid_steps", 0) + 1
            else:
                state_info["valid_tool_calls"] = state_info.get("valid_tool_calls", 0) + 1
            state_info["last_tool_result"] = tool_result

    submit_payloads = _extract_tag_payload(SUBMIT_PATTERN, model_response)
    if submit_payloads:
        target = doc["target_state"]
        success = state.get("cash", 0) >= target["min_cash"] and state.get("days_elapsed", 0) >= target["min_days_elapsed"]
        final_payload = {
            "success": success,
            "tool_calls": state_info["tool_calls"],
            "valid_tool_calls": state_info.get("valid_tool_calls", 0),
            "invalid_steps": state_info.get("invalid_steps", 0),
            "cash": state.get("cash", 0.0),
            "days_elapsed": state.get("days_elapsed", 0),
            "inventory": state.get("inventory", 0),
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
    state_info["last_tool_result"] = no_action_result
    next_prompt = _build_agent_prompt(doc, state, tool_result=no_action_result)
    return [], next_prompt, False, previous_output, state_info


def vending_process_results(doc, results):
    raw = results[0] if results else ""
    success = 0.0
    final_balance = 0.0
    days_active = 0.0
    trace_step_validity = 0.0
    trace_state_progress = 0.0
    trace_termination_quality = 0.0
    trace_quality = 0.0

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            success = 1.0 if payload.get("success") else 0.0
            final_balance = float(payload.get("cash", 0.0))
            days_active = float(payload.get("days_elapsed", 0.0))

            valid_calls = float(payload.get("valid_tool_calls", payload.get("tool_calls", 0.0)))
            invalid_steps = float(payload.get("invalid_steps", 0.0))
            denom = max(valid_calls + invalid_steps, 1.0)
            trace_step_validity = valid_calls / denom

            target = doc.get("target_state", {})
            cash_target = float(target.get("min_cash", 0.0))
            day_target = float(target.get("min_days_elapsed", 0.0))
            cash_progress = min(final_balance / cash_target, 1.0) if cash_target > 0 else 1.0
            day_progress = min(days_active / day_target, 1.0) if day_target > 0 else 1.0
            trace_state_progress = (cash_progress + day_progress) / 2.0

            trace_termination_quality = 1.0 if payload.get("submit") else 0.0
            trace_quality = (trace_step_validity + trace_state_progress + trace_termination_quality) / 3.0
        except json.JSONDecodeError:
            success = 0.0
            final_balance = 0.0
            days_active = 0.0

    return {
        "vending_success": success,
        "vending_final_balance": final_balance,
        "vending_days_active": days_active,
        "vending_trace_quality": trace_quality,
        "vending_trace_step_validity": trace_step_validity,
        "vending_trace_state_progress": trace_state_progress,
        "vending_trace_termination_quality": trace_termination_quality,
    }


def vending_aggregate_success(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def vending_aggregate_final_balance(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def vending_aggregate_days_active(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def vending_aggregate_trace_quality(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def vending_aggregate_trace_step_validity(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def vending_aggregate_trace_state_progress(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def vending_aggregate_trace_termination_quality(results):
    if not results:
        return 0.0
    return sum(results) / len(results)
