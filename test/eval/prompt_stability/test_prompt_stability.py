"""Prompt Stability Tests - Version Check for Classic Benchmarks

Catches silent prompt drift in evaluation tasks. For each task we freeze a
sample document (with its doc_id) and snapshot the exact prompt text that
would be sent to the model under the *default* lmms_eval_specific_kwargs.

Any change to the prompt construction pipeline -- system prompt wording,
format requirements, option formatting, pre/post prompt -- will fail these
tests, forcing explicit review.

    # Regenerate snapshots after an INTENTIONAL prompt change:
    pytest test/eval/prompt_stability/ --update-snapshots -v

Covered tasks (8 benchmarks, 11 variants):
    General       : MMMU (mc / open), MMBench (hint / no-hint)
    Hallucination : HallusionBench
    Document      : OCRBench, MMLongBench-Doc
    Spatial       : VSI-Bench (mca / na)
    Vision-centric: MMVP, BLINK
"""

import json
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


# ---------------------------------------------------------------------------
# Lazy importers -- isolate module-level side-effects
# ---------------------------------------------------------------------------
def _import_mmmu():
    """MMMU utils creates an LLM-judge server at module level; mock it."""
    with mock.patch("lmms_eval.llm_judge.get_server", return_value=mock.MagicMock()):
        from lmms_eval.tasks.mmmu.utils import mmmu_doc_to_text
    return mmmu_doc_to_text


def _import_mmbench():
    from lmms_eval.tasks.mmbench.en_utils import mmbench_doc_to_text

    return mmbench_doc_to_text


def _import_hallusion():
    from lmms_eval.tasks.hallusion_bench.evaluate_hb import hb_doc_to_text

    return hb_doc_to_text


def _import_ocrbench():
    from lmms_eval.tasks.ocrbench.utils import ocrbench_doc_to_text

    return ocrbench_doc_to_text


def _import_mmlongbench():
    from lmms_eval.tasks.mmlongbench_doc.utils import mmlongbench_doc_to_text

    return mmlongbench_doc_to_text


def _import_vsibench():
    from lmms_eval.tasks.vsibench.utils import vsibench_doc_to_text

    return vsibench_doc_to_text


def _import_mmvp():
    from lmms_eval.tasks.mmvp.utils import mmvp_doc_to_text

    return mmvp_doc_to_text


def _import_blink():
    from lmms_eval.tasks.blink.utils import blink_doc_to_text

    return blink_doc_to_text


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _doc_id(doc: dict) -> str:
    """Extract the most natural identifier from a fixture doc."""
    for key in ("id", "doc_id", "idx", "Index", "index"):
        if key in doc:
            return str(doc[key])
    return "unknown"


# ===================================================================
# Task registry -- one entry per (task, variant) snapshot
#
# Each entry carries:
#   get_fn          callable returning the real doc_to_text function
#   default_kwargs  the lmms_eval_specific_kwargs.default from YAML
#   gen_kwargs      the generation_kwargs from YAML
#   fixture         a minimal doc dict (text fields only, no images)
# ===================================================================

CASES: dict = {
    # ------------------------------------------------------------------
    # MMMU  (General)
    # ------------------------------------------------------------------
    "mmmu_val__mc": {
        "task": "mmmu_val",
        "variant": "multiple-choice",
        "get_fn": _import_mmmu,
        "default_kwargs": {
            "prompt_type": "format",
            "multiple_choice_prompt": "Answer with the option's letter from the given choices directly.",
            "open_ended_prompt": "Answer the question using a single word or phrase.",
        },
        "gen_kwargs": {"max_new_tokens": 128},
        "fixture": {
            "id": "validation_Accounting_1",
            "question": "In the context of agency theory, which scenario best illustrates a Type I agency problem?",
            "question_type": "multiple-choice",
            "options": "['Shareholders disagreeing with bondholders over dividend policy', "
            "'Managers pursuing personal projects that reduce firm value', "
            "'Majority shareholders expropriating minority shareholders', "
            "'Suppliers refusing to extend credit terms to the firm']",
            "answer": "B",
        },
    },
    "mmmu_val__open": {
        "task": "mmmu_val",
        "variant": "open-ended",
        "get_fn": _import_mmmu,
        "default_kwargs": {
            "prompt_type": "format",
            "multiple_choice_prompt": "Answer with the option's letter from the given choices directly.",
            "open_ended_prompt": "Answer the question using a single word or phrase.",
        },
        "gen_kwargs": {"max_new_tokens": 128},
        "fixture": {
            "id": "validation_Math_30",
            "question": "What is the area of the shaded region in the figure shown?",
            "question_type": "open",
            "options": "[]",
            "answer": "16",
        },
    },
    # ------------------------------------------------------------------
    # MMBench  (General)
    # ------------------------------------------------------------------
    "mmbench_en_dev__hint": {
        "task": "mmbench_en_dev",
        "variant": "with-hint",
        "get_fn": _import_mmbench,
        "default_kwargs": {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        },
        "gen_kwargs": {
            "until": ["ASSISTANT:"],
            "max_new_tokens": 1024,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "What activity is the person in the image performing?",
            "hint": "The person is outdoors near a body of water.",
            "answer": "C",
            "A": "Swimming",
            "B": "Running",
            "C": "Fishing",
            "D": "Cycling",
            "index": 101,
            "source": "internet",
            "split": "dev",
            "category": "Activity Recognition",
            "L2-category": "Activity",
        },
    },
    "mmbench_en_dev__nohint": {
        "task": "mmbench_en_dev",
        "variant": "no-hint",
        "get_fn": _import_mmbench,
        "default_kwargs": {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        },
        "gen_kwargs": {
            "until": ["ASSISTANT:"],
            "max_new_tokens": 1024,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "What color is the car in the image?",
            "hint": "nan",
            "answer": "B",
            "A": "Red",
            "B": "Blue",
            "C": "Green",
            "D": "Yellow",
            "index": 202,
            "source": "internet",
            "split": "dev",
            "category": "Attribute Recognition",
            "L2-category": "Color",
        },
    },
    # ------------------------------------------------------------------
    # HallusionBench  (Hallucination)
    # ------------------------------------------------------------------
    "hallusion_bench_image": {
        "task": "hallusion_bench_image",
        "variant": None,
        "get_fn": _import_hallusion,
        "default_kwargs": {"pre_prompt": "", "post_prompt": ""},
        "gen_kwargs": {
            "max_new_tokens": 128,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "Is there a cat sitting on the table in this image?",
            "gt_answer_details": "No, there is no cat on the table.",
            "category": "VD",
            "subcategory": "object_attribute",
            "set_id": 1,
            "figure_id": 1,
            "question_id": 1,
        },
    },
    # ------------------------------------------------------------------
    # OCRBench  (Document)
    # ------------------------------------------------------------------
    "ocrbench": {
        "task": "ocrbench",
        "variant": None,
        "get_fn": _import_ocrbench,
        "default_kwargs": {"pre_prompt": "", "post_prompt": ""},
        "gen_kwargs": {
            "max_new_tokens": 128,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "What is the text written on the sign in this image?",
            "answer": "STOP",
            "question_type": "Regular Text Recognition",
            "dataset": "IIIT5K",
        },
    },
    # ------------------------------------------------------------------
    # MMLongBench-Doc  (Document)
    # ------------------------------------------------------------------
    "mmlongbench_doc": {
        "task": "mmlongbench_doc",
        "variant": None,
        "get_fn": _import_mmlongbench,
        "default_kwargs": {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with a single word or phrase. If the answer is not in the document, answer 'Not answerable'.",
        },
        "gen_kwargs": {
            "max_new_tokens": 64,
            "temperature": 0,
            "do_sample": False,
        },
        "fixture": {
            "question": "What is the total revenue reported in Q3 2023?",
            "answer": "$4.2 billion",
            "answer_format": "Str",
            "doc_id": "annual_report_2023.pdf",
            "evidence_pages": "[12]",
        },
    },
    # ------------------------------------------------------------------
    # VSI-Bench  (Spatial)
    # ------------------------------------------------------------------
    "vsibench__mca": {
        "task": "vsibench",
        "variant": "multiple-choice",
        "get_fn": _import_vsibench,
        "default_kwargs": {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Please answer the question using a single word or phrase.",
        },
        "gen_kwargs": {
            "max_new_tokens": 16,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "Which direction is the sofa relative to the dining table?",
            "question_type": "object_rel_direction_easy",
            "options": ["To the left", "To the right", "In front", "Behind"],
            "ground_truth": "A",
            "dataset": "scannet",
            "scene_name": "scene0001_00",
        },
    },
    "vsibench__na": {
        "task": "vsibench",
        "variant": "numerical-answer",
        "get_fn": _import_vsibench,
        "default_kwargs": {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Please answer the question using a single word or phrase.",
        },
        "gen_kwargs": {
            "max_new_tokens": 16,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "How many chairs are visible in this room?",
            "question_type": "object_counting",
            "options": [],
            "ground_truth": "4",
            "dataset": "scannet",
            "scene_name": "scene0001_00",
        },
    },
    # ------------------------------------------------------------------
    # MMVP  (Vision-centric)
    # ------------------------------------------------------------------
    "mmvp": {
        "task": "mmvp",
        "variant": None,
        "get_fn": _import_mmvp,
        "default_kwargs": {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        },
        "gen_kwargs": {
            "max_new_tokens": 16,
            "temperature": 0,
            "do_sample": False,
        },
        "fixture": {
            "Question": "Is the arrow pointing up or down?",
            "Options": "(a) Up (b) Down",
            "Correct Answer": "(a)",
            "Index": 1,
        },
    },
    # ------------------------------------------------------------------
    # BLINK  (Vision-centric)
    # ------------------------------------------------------------------
    "blink__art_style": {
        "task": "blink_art_style",
        "variant": None,
        "get_fn": _import_blink,
        "default_kwargs": {
            "pre_prompt": "Note: You only need to respond with {} without providing any additional information.\n",
            "post_prompt": "",
        },
        "gen_kwargs": {"max_new_tokens": 1024},
        "fixture": {
            "prompt": "Which of the following art styles best matches the painting shown in the image?",
            "choices": ["Impressionism", "Cubism", "Baroque", "Pop Art"],
            "answer": "(A)",
            "idx": 1,
            "sub_task": "Art_Style",
        },
    },
}


# ===================================================================
# Tests
# ===================================================================


@pytest.mark.parametrize("case_name", sorted(CASES.keys()))
def test_prompt_stable(case_name, update_snapshots):
    """The prompt sent to the model must not change between versions."""
    case = CASES[case_name]

    # --- call the real doc_to_text ---
    doc_to_text = case["get_fn"]()
    prompt = doc_to_text(case["fixture"], case["default_kwargs"])

    snapshot_path = SNAPSHOT_DIR / f"{case_name}.json"

    # --- update mode: write snapshot and skip ---
    if update_snapshots:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "task": case["task"],
            "variant": case.get("variant"),
            "doc_id": _doc_id(case["fixture"]),
            "prompt_text": prompt,
            "gen_kwargs": case["gen_kwargs"],
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
        pytest.skip("snapshot updated")

    # --- compare mode ---
    assert snapshot_path.exists(), f"No snapshot found for '{case_name}'.\n" f"Generate with:  pytest test/eval/prompt_stability/ --update-snapshots -v"

    expected = json.loads(snapshot_path.read_text())

    # 1) prompt text must match exactly
    assert prompt == expected["prompt_text"], (
        f"Prompt changed for '{case_name}'!\n" f"If intentional, run:\n" f"  pytest test/eval/prompt_stability/ --update-snapshots -v\n\n" f"--- Expected ---\n{expected['prompt_text']}\n\n" f"--- Got ---\n{prompt}"
    )

    # 2) gen_kwargs must match
    assert case["gen_kwargs"] == expected["gen_kwargs"], f"gen_kwargs changed for '{case_name}'!\n" f"Expected: {json.dumps(expected['gen_kwargs'], indent=2)}\n" f"Got:      {json.dumps(case['gen_kwargs'], indent=2)}"


@pytest.mark.parametrize("case_name", sorted(CASES.keys()))
def test_gen_kwargs_complete(case_name):
    """Every task must specify max_new_tokens in gen_kwargs."""
    case = CASES[case_name]
    assert "max_new_tokens" in case["gen_kwargs"], f"max_new_tokens missing from gen_kwargs for '{case_name}'"
