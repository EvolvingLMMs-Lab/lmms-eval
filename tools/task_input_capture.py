#!/usr/bin/env python3
"""Capture task request/model-input payloads for deterministic A/B comparison.

This tool supports two modes from a YAML spec:

1) `cases`: direct function calls (legacy/manual mode)
2) `tasks`: real-task mode that loads real datasets via TaskManager, builds
   requests with `build_all_requests`, then captures boundary payloads that will
   be fed into model methods.

The CI use-case is mode (2): compare base vs head on representative tasks
without running real model generation.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def _build_image(spec: dict[str, Any]) -> Image.Image:
    size = spec.get("size", [2, 2])
    mode = spec.get("mode", "RGB")
    color = spec.get("color", [255, 255, 255])
    return Image.new(mode, tuple(size), tuple(color))


def _materialize(value: Any) -> Any:
    if isinstance(value, dict):
        if "__image__" in value:
            return _build_image(value["__image__"])
        return {k: _materialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize(v) for v in value]
    return value


def _normalize(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return {
            "__type__": "PIL.Image",
            "mode": value.mode,
            "size": [value.size[0], value.size[1]],
        }
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return {"__repr__": repr(value), "__type__": type(value).__name__}


def _reset_lmms_eval_modules() -> None:
    for name in list(sys.modules.keys()):
        if name == "lmms_eval" or name.startswith("lmms_eval."):
            del sys.modules[name]


def _import_function(repo_root: Path, function_path: str):
    _reset_lmms_eval_modules()
    sys.path.insert(0, str(repo_root))
    try:
        module_name, fn_name = function_path.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, fn_name)
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)


def _callable_descriptor(fn: Any) -> dict[str, Any]:
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    return {"module": module, "qualname": qualname}


def _summarize_messages(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"messages": [], "text_segments": []}

    normalized_messages: list[dict[str, Any]] = []
    text_segments: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append({"role": "", "content": []})
            continue

        role = str(message.get("role", ""))
        content = message.get("content")
        normalized_content: list[dict[str, Any]] = []

        if isinstance(content, str):
            text_value = str(content)
            normalized_content.append({"type": "text", "text": text_value})
            text_segments.append(text_value)
            normalized_messages.append({"role": role, "content": normalized_content})
            continue

        if not isinstance(content, list):
            normalized_messages.append({"role": role, "content": []})
            continue

        for item in content:
            if not isinstance(item, dict):
                normalized_content.append({"type": ""})
                continue

            item_type = str(item.get("type", ""))
            if item_type == "text":
                text_value = str(item.get("text", ""))
                normalized_content.append({"type": "text", "text": text_value})
                text_segments.append(text_value)
            else:
                normalized_content.append({"type": item_type})

        normalized_messages.append({"role": role, "content": normalized_content})

    return {
        "messages": normalized_messages,
        "text_segments": text_segments,
    }


def _load_task_object(repo_root: Path, task_name: str):
    _reset_lmms_eval_modules()
    sys.path.insert(0, str(repo_root))
    try:
        from lmms_eval import utils as lmms_utils
        from lmms_eval.tasks import TaskManager, get_task_dict

        manager = TaskManager(include_defaults=True)
        yaml_path = manager.task_index[task_name]["yaml_path"]
        task_cfg = lmms_utils.load_yaml_config(yaml_path, mode="full") if yaml_path != -1 else {}

        use_chat = bool(task_cfg.get("doc_to_messages", None)) and task_cfg.get("output_type") == "generate_until"
        task_type = "chat" if use_chat else "simple"

        loaded = get_task_dict([task_name], task_manager=manager, task_type=task_type)
        return loaded[task_name]
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)


def _capture_instance_boundary(task_obj, instance, *, capture_mode: str) -> dict[str, Any]:
    args = instance.args
    request_type = instance.request_type
    run_callables = capture_mode in {"text_and_structure", "full"}
    capture_visual = capture_mode == "full"

    payload: dict[str, Any] = {
        "request_type": request_type,
        "idx": instance.idx,
        "doc_id": _to_jsonable(instance.doc_id),
    }

    if request_type == "generate_until":
        if len(args) == 6 and callable(args[1]):
            ctx, doc_to_messages, gen_kwargs, doc_id, _, split = args
            payload.update(
                {
                    "ctx": _normalize(ctx),
                    "generation_kwargs": _normalize(copy.deepcopy(gen_kwargs)),
                    "doc_to_messages_callable": _callable_descriptor(doc_to_messages),
                }
            )
            if run_callables:
                doc = task_obj.dataset[split][doc_id]
                messages = doc_to_messages(doc)
                payload["doc_to_messages_summary"] = _normalize(_summarize_messages(messages))
        elif len(args) == 6:
            ctx, gen_kwargs, doc_to_visual, doc_id, _, split = args
            payload.update(
                {
                    "ctx": _normalize(ctx),
                    "generation_kwargs": _normalize(copy.deepcopy(gen_kwargs)),
                    "doc_to_visual_callable": _callable_descriptor(doc_to_visual) if callable(doc_to_visual) else {"value": _normalize(doc_to_visual)},
                }
            )
            if capture_visual and callable(doc_to_visual):
                doc = task_obj.dataset[split][doc_id]
                payload["doc_to_visual_output"] = _normalize(doc_to_visual(doc))
        else:
            payload["raw_args"] = _normalize(args)

    elif request_type == "generate_until_multi_round":
        if len(args) == 7:
            ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, _, split = args
            payload.update(
                {
                    "ctx": _normalize(ctx),
                    "generation_kwargs": _normalize(copy.deepcopy(gen_kwargs)),
                    "doc_to_visual_callable": _callable_descriptor(doc_to_visual) if callable(doc_to_visual) else {"value": _normalize(doc_to_visual)},
                    "doc_to_text_callable": _callable_descriptor(doc_to_text) if callable(doc_to_text) else {"value": _normalize(doc_to_text)},
                }
            )
            if run_callables:
                doc = task_obj.dataset[split][doc_id]
                if callable(doc_to_text):
                    payload["doc_to_text_output"] = _normalize(doc_to_text(doc))
                if capture_visual and callable(doc_to_visual):
                    payload["doc_to_visual_output"] = _normalize(doc_to_visual(doc))
        else:
            payload["raw_args"] = _normalize(args)

    elif request_type == "loglikelihood":
        if len(args) >= 6:
            ctx, continuation_or_fn, doc_to_visual, doc_id, _, split = args[:6]
            if not callable(continuation_or_fn):
                continuation_value = continuation_or_fn
            elif run_callables or capture_mode == "request_only":
                continuation_value = continuation_or_fn(task_obj.dataset[split][doc_id])
            else:
                continuation_value = _callable_descriptor(continuation_or_fn)

            payload.update(
                {
                    "ctx": _normalize(ctx),
                    "continuation": _normalize(continuation_value),
                    "doc_to_visual_callable": _callable_descriptor(doc_to_visual) if callable(doc_to_visual) else {"value": _normalize(doc_to_visual)},
                }
            )
            if capture_visual and callable(doc_to_visual):
                doc = task_obj.dataset[split][doc_id]
                payload["doc_to_visual_output"] = _normalize(doc_to_visual(doc))
        else:
            payload["raw_args"] = _normalize(args)
    else:
        payload["raw_args"] = _normalize(args)

    return payload


def _instance_sort_key(instance):
    def _norm(v):
        try:
            return (0, int(v))
        except Exception:
            return (1, str(v))

    return (_norm(instance.doc_id), _norm(instance.idx))


def capture_tasks(spec_path: Path, repo_root: Path) -> dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    tasks = spec.get("tasks", [])
    output: dict[str, Any] = {"tasks": []}

    for task_case in tasks:
        case_id = task_case["id"]
        task_name = task_case["task"]
        limit = int(task_case.get("limit", 1))
        max_requests = int(task_case.get("max_requests", 1))
        capture_mode = task_case.get("capture_mode", "request_only")
        allowed_modes = {"request_only", "text_and_structure", "full"}
        if capture_mode not in allowed_modes:
            raise ValueError(f"Unsupported capture_mode: {capture_mode}. Allowed: {sorted(allowed_modes)}")

        task_obj = _load_task_object(repo_root, task_name)
        task_obj.build_all_requests(
            limit=limit,
            offset=0,
            rank=0,
            world_size=1,
            cache_requests=False,
            rewrite_requests_cache=False,
            system_instruction=None,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            chat_template=None,
            tokenizer_name="",
        )

        instances = sorted(task_obj.instances, key=_instance_sort_key)
        captured = [_capture_instance_boundary(task_obj, inst, capture_mode=capture_mode) for inst in instances[:max_requests]]

        output["tasks"].append(
            {
                "id": case_id,
                "task": task_name,
                "limit": limit,
                "capture_mode": capture_mode,
                "captured_requests": captured,
            }
        )

    output["tasks"].sort(key=lambda x: x["id"])
    return output


def capture(spec_path: Path, repo_root: Path) -> dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if "tasks" in spec:
        return capture_tasks(spec_path=spec_path, repo_root=repo_root)

    cases = spec.get("cases", [])
    output: dict[str, Any] = {"cases": []}

    for case in cases:
        case_id = case["id"]
        function_path = case["function"]
        kwargs = _materialize(case.get("kwargs", {}))

        fn = _import_function(repo_root, function_path)
        result = fn(**kwargs)
        output["cases"].append(
            {
                "id": case_id,
                "function": function_path,
                "kwargs": _normalize(kwargs),
                "output": _normalize(result),
            }
        )

    output["cases"].sort(key=lambda x: x["id"])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture normalized task model-input snapshots")
    parser.add_argument("--spec", required=True, help="YAML spec file with capture cases")
    parser.add_argument("--repo-root", default=".", help="Target repository root to import from")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--compare-with", default=None, help="Optional baseline JSON to compare against")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    snapshot = capture(spec_path=spec_path, repo_root=repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    if args.compare_with:
        compare_path = Path(args.compare_with).resolve()
        baseline = json.loads(compare_path.read_text(encoding="utf-8"))
        if baseline != snapshot:
            print("Task input snapshot mismatch detected.")
            print(f"current: {output_path}")
            print(f"baseline: {compare_path}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
