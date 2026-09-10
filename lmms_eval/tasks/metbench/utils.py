"""Dataset preparation and chat hooks for the MET-Bench tasks."""

from typing import Any

from datasets import Dataset

from lmms_eval.tasks.metbench.metbench_core import build_messages, score


def _prepare(dataset: Dataset, domain: str) -> Dataset:
    """Attach scoring fields without decoding the fixed split's images."""
    if len(dataset) != 500:
        raise ValueError("Expected the released 500-example evaluation split")
    target = "correct_choice" if domain == "minecraft" else "final_state"
    return dataset.add_column("metbench_domain", [domain] * len(dataset)).add_column("target", dataset[target])


def prepare_chess(dataset: Dataset) -> Dataset:
    """Prepare the fixed Chess test split."""
    return _prepare(dataset, "chess")


def prepare_shell(dataset: Dataset) -> Dataset:
    """Prepare the fixed Shell Game test split."""
    return _prepare(dataset, "shell")


def prepare_minecraft(dataset: Dataset) -> Dataset:
    """Prepare the fixed Minecraft test split."""
    return _prepare(dataset, "minecraft")


def doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Return the text portions of the selected modality for logging and context."""
    return "".join(part["text"] for part in doc_to_messages(doc, lmms_eval_specific_kwargs)[0]["content"] if part["type"] == "text")


def doc_to_visual(doc: dict[str, Any]) -> list:
    """Require a chat backend to preserve the interleaved image prompts."""
    raise ValueError("MET-Bench image tasks require an lmms-eval chat backend")


def doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> list[dict[str, Any]]:
    """Construct the benchmark's interleaved chat prompt."""
    return build_messages(doc, (lmms_eval_specific_kwargs or {}).get("modality", "text"))


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Return one score per example; Chess squares stay within their board."""
    return {"acc": score(doc, results[0])}
