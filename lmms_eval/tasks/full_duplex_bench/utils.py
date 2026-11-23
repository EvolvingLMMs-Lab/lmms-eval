"""Full-Duplex-Bench v1.5 utilities for lmms-eval.

Full-Duplex-Bench (https://github.com/DanielLin94144/Full-Duplex-Bench,
v1.5 paper: arXiv:2507.23159) evaluates *overlap handling* in full-duplex
spoken dialogue models. Each sample contains a "clean" conversation audio and
a "noisy" variant that injects an overlap (user interruption, backchannel,
side conversation, or background speech). A duplex wrapper runs the model on
both variants as independent sessions, and this module scores the result with
the official v1.5 pipeline:

1. build word-level timestamped transcripts for the model outputs (the duplex
   wrapper reports text per streaming chunk together with the engine clock),
2. send {input_clean, input_noisy, output_clean, output_noisy} to an
   OpenAI-compatible judge with the official `behavior.txt` instruction,
3. map the returned C-axis label to the per-task correct label
   (user_interruption -> C_RESPOND, everything else -> C_RESUME).

The automatic timing metric `fdb_responded_after_overlap` additionally checks
that the model produced speech after the overlap without any judge.
"""

import json
import os
import re
from typing import Any, Dict, List

from loguru import logger as eval_logger

# Official v1.5 judge instruction (instruction/behavior.txt), verbatim.
BEHAVIOR_INSTRUCTION = """Goal
------------
After a overlap begins in the user audio, label how the model handles
that interruption in its subsequent speech. Assign **exactly one C_* tag**
(content relation).


Input (JSON)
------------
Expected fields (arrays of word-level dicts with timestamps in seconds):

{
  "input_clean":  [...],   // user words before any overlap
  "input_noisy":  [...],   // user words + overlap words
  "output_clean": [...],   // model reply to clean input
  "output_noisy": [...]    // model reply when distracted
}
Each element: {"text": "<word>", "timestamp": [start, end]}  // seconds


Required Output
---------------
Produce **one single line**:

{ "behaviour": ["C_*"] }

• Replace C_* with one tag from the list below.
• If the model never speaks after the overlap: use "C_UNKNOWN".
• Absolutely no characters — not even a newline — may follow the closing brace.


Distractor Reference Points (for internal reasoning only)
---------------------------------------------------------
You may consult timestamps but do NOT output them.

overlap_start  = start time of the first overlap word in input_noisy
overlap_end    = end time of the last overlap word in input_noisy


Semantic-Segment Rule  (retain as given)
----------------------------------------
Open a *new segment* when—after overlap_start—one of the following happens:
  1. The model directly addresses the noise (apology, request to repeat…).
  2. The dialog act changes (answer → apology, answer → question…).
  3. A meta-discourse opener (“Anyway,” “To continue,” …) appears.
  4. Clear semantic reset; punctuation alone is not enough.

Segments that begin *before* overlap_start do **not** qualify, even if words
overlap with the overlap.

Use this rule to decide which stretch of model speech counts as the first
meaningful post-overlap segment that you will label with a C_* tag. If no
new segment occurs, choose C_UNKNOWN.


C-Axis: Content Relation
------------------------
Judge the first *valid* post-overlap segment identified via the rule above.
Choose exactly one label:

C_RESPOND
    The model meaningfully addresses the overlap's content (answers,
    clarifies, or reacts to what the overlap said / introduced).

C_RESUME
    The model ignores the overlap and continues or completes the
    pre-overlap task / answer that was already in progress.

C_UNCERTAIN_HANDLING
    The model signals uncertainty or difficulty hearing / missing info
    (“I didn't catch that…”, “Could you repeat?”) without giving a clear,
    content-specific answer. (All generic repeat/clarification moves fall here.)

C_UNKNOWN
    The model output is semantically off-target or low-information: does
    not clearly resume, respond, or express uncertainty (e.g., unrelated filler,
    templated noise). Includes cases where there is no model speech at all
    after the overlap.


Annotation Procedure
--------------------
1. Locate overlap_start / overlap_end in input_noisy (for reference).
2. Scan output_noisy for model speech occurring after overlap_start.
3. Using the Semantic-Segment Rule, find the first *new* post-overlap
   semantic segment. Ignore stretches that began before the overlap.
4. If no new segment: label C_UNKNOWN.
5. Otherwise assign the single best C_* tag to that segment.
6. Optionally write brief reasoning comments.
7. Output the required single-line JSON exactly as specified above (no newline)."""


# ---------------------------------------------------------------------------
# Chunk -> word transcript conversion
# ---------------------------------------------------------------------------


def chunks_to_word_transcript(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert duplex streaming chunks into the official word-level format.

    Each duplex chunk carries the text emitted during one streaming step and
    the engine clock at that step; the words are spread evenly over the
    chunk's one-second span.
    """
    words: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("text", "") or ""
        text = text.strip()
        if not text:
            continue
        end_time = float(chunk.get("current_time") or 0.0)
        start_time = max(0.0, end_time - 1.0)
        tokens = [t for t in re.split(r"\s+", text) if t]
        if not tokens:
            continue
        span = (end_time - start_time) / len(tokens)
        for i, token in enumerate(tokens):
            word_start = start_time + i * span
            words.append({"text": token, "timestamp": [round(word_start, 2), round(word_start + span, 2)]})
    return words


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Judge client (OpenAI-compatible, official eval_behavior.py protocol)
# ---------------------------------------------------------------------------


def _judge_env() -> Dict[str, str]:
    return {
        "api_key": os.environ.get("FDB_JUDGE_API_KEY") or os.environ.get("DUPLEXEVAL_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("FDB_JUDGE_BASE_URL") or os.environ.get("DUPLEXEVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL", ""),
        "model": os.environ.get("FDB_JUDGE_MODEL") or os.environ.get("DUPLEXEVAL_MODEL") or os.environ.get("OPENAI_MODEL_ID", ""),
    }


def judge_behaviour(payload: Dict[str, Any]) -> List[str]:
    import requests

    env = _judge_env()
    if not env["api_key"] or not env["base_url"] or not env["model"]:
        raise ValueError("Full-Duplex-Bench behavior judging needs an OpenAI-compatible endpoint: set FDB_JUDGE_API_KEY / FDB_JUDGE_BASE_URL / FDB_JUDGE_MODEL (or the DUPLEXEVAL_* / OPENAI_* fallbacks).")

    response = requests.post(
        f"{env['base_url'].rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {env['api_key']}", "Content-Type": "application/json"},
        json={
            "model": env["model"],
            "messages": [
                {"role": "system", "content": BEHAVIOR_INSTRUCTION},
                {"role": "user", "content": payload},
            ],
            "temperature": 0.1,
        },
        timeout=180,
    )
    response.raise_for_status()
    prediction = response.json()["choices"][0]["message"]["content"]

    decoder = json.JSONDecoder()
    pos = prediction.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(prediction, pos)
            if "behaviour" in obj:
                return list(obj["behaviour"])
            pos = prediction.find("{", end)
        except json.JSONDecodeError:
            pos = prediction.find("{", pos + 1)
    raise ValueError(f"No JSON object with key 'behaviour' found in judge reply: {prediction[:200]}")


# ---------------------------------------------------------------------------
# Doc accessors
# ---------------------------------------------------------------------------


def fdb_doc_to_visual(doc: dict) -> List[str]:
    """Two audio streams in official order: clean session, then overlapped."""
    return [doc["clean_input_wav"], doc["noisy_input_wav"]]


def fdb_doc_to_text(doc: dict) -> str:
    return f"Full-Duplex-Bench v1.5 {doc['fdb_task']} sample {doc['sample_id']}"


def _overlap_start(doc: dict) -> float:
    timestamps = doc.get("timestamps") or []
    if timestamps and isinstance(timestamps[0], (int, float)):
        return float(timestamps[0])
    return 0.0


def _parse_sessions(raw: str) -> List[List[Dict[str, Any]]]:
    try:
        payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except json.JSONDecodeError:
        return [[], []]
    sessions = payload.get("sessions", [])
    while len(sessions) < 2:
        sessions.append([])
    return sessions[:2]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def fdb_process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    clean_chunks, noisy_chunks = _parse_sessions(results[0] if results else "[]")
    output_clean = chunks_to_word_transcript(clean_chunks)
    output_noisy = chunks_to_word_transcript(noisy_chunks)

    overlap_start = _overlap_start(doc)
    responded = any(not chunk.get("is_listen", False) and (chunk.get("text") or "").strip() and float(chunk.get("current_time") or 0.0) >= overlap_start for chunk in noisy_chunks)

    payload = _compact(
        {
            "input_clean": doc.get("input_clean_words", []),
            "input_noisy": doc.get("input_noisy_words", []),
            "output_clean": output_clean,
            "output_noisy": output_noisy,
        }
    )

    label = ""
    try:
        labels = judge_behaviour(payload)
        label = labels[0] if labels else ""
    except Exception as exc:
        eval_logger.error(f"FDB judge failed for {doc.get('sample_id')}: {exc}")

    return {
        "fdb_behavior_acc": float(label == doc["correct_label"]) if label else 0.0,
        "fdb_responded_after_overlap": float(responded),
        "fdb_label_correct": float(label == doc["correct_label"]) if label else 0.0,
    }


def fdb_aggregate_accuracy(results: List[float]) -> float:
    values = [float(value) for value in results if value is not None]
    return sum(values) / len(values) if values else 0.0


def fdb_aggregate_responded(results: List[float]) -> float:
    values = [float(value) for value in results if value is not None]
    return sum(values) / len(values) if values else 0.0
