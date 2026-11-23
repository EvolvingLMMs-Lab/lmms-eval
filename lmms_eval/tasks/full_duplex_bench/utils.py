"""Full-Duplex-Bench v1.5 utilities for lmms-eval.

Full-Duplex-Bench (https://github.com/DanielLin94144/Full-Duplex-Bench,
v1.5 paper: arXiv:2507.23159) evaluates *overlap handling* in full-duplex
spoken dialogue models. Each sample contains a "clean" conversation audio and
a "noisy" variant that injects an overlap (user interruption, backchannel,
side conversation, or background speech). A duplex wrapper runs the model on
both variants as independent sessions, and this module scores the result with
the official v1.5 behaviour judge (v1_v1.5/evaluation/eval_behavior.py):

1. build ``{"text", "chunks"}`` word-level transcripts for the model outputs,
2. send {input_clean, input_noisy, output_clean, output_noisy} to an
   OpenAI-compatible judge with the official ``behavior.txt`` instruction
   (default judge: the official ``gpt-4o-2024-08-06``),
3. report the share of each C-axis tag, as the official script does, plus
   ``fdb_expected_behavior``: the share of samples whose tag is an expected
   behaviour for the task (paper Sec. 2: Respond for user interruption; Resume
   for backchannel; Resume or Uncertain for talking to others and background
   speech).

Deviation from the official pipeline: the official evaluation runs ASR with
word timestamps (parakeet-tdt-0.6b-v2) on the model's output audio. Here the
output transcripts come from the text that the duplex model emits per
one-second chunk, with the words spread evenly over that second, so word
timing is approximate. Scores are therefore not directly comparable with the
paper's tables.

Samples that cannot be scored (generation error, or a judge that still fails
after ``FDB_JUDGE_MAX_ATTEMPTS`` tries) are left out of the label shares and
counted in ``fdb_scoring_failed``.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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
    chunk's one-second span. This approximates the official pipeline, which
    runs ASR with word timestamps on the model's output audio.
    """
    words: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("text", "") or ""
        text = text.strip()
        if not text:
            continue
        current_time = chunk.get("current_time")
        if current_time is None:
            # the engine clock ends each one-second chunk at chunk_idx + 1
            current_time = int(chunk.get("chunk_idx", 0)) + 1
        end_time = float(current_time)
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


# The official judge (v1_v1.5/evaluation/eval_behavior.py).
OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"
OPENAI_BASE_URL = "https://api.openai.com/v1"
# The official loop retries API errors and unparsable replies (seed + 1, 5 s
# pause) without limit; here the number of attempts is bounded.
JUDGE_RETRY_SECONDS = 5.0
# HTTP statuses that mean the endpoint or the key is wrong, not that one call
# failed: they stop the run instead of being retried.
_FATAL_HTTP_STATUSES = (401, 403, 404)


class JudgeConfigError(RuntimeError):
    """The judge endpoint cannot be used at all (missing or rejected credentials)."""


def _judge_env() -> Dict[str, str]:
    env = {
        "api_key": os.environ.get("FDB_JUDGE_API_KEY") or os.environ.get("DUPLEXEVAL_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("FDB_JUDGE_BASE_URL") or os.environ.get("DUPLEXEVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL,
        "model": os.environ.get("FDB_JUDGE_MODEL") or os.environ.get("DUPLEXEVAL_MODEL") or os.environ.get("OPENAI_MODEL_ID") or OFFICIAL_JUDGE_MODEL,
    }
    if not env["api_key"]:
        raise JudgeConfigError(
            "Full-Duplex-Bench behavior judging needs an OpenAI-compatible endpoint: set FDB_JUDGE_API_KEY "
            "(optionally FDB_JUDGE_BASE_URL and FDB_JUDGE_MODEL; default model is the official "
            f"{OFFICIAL_JUDGE_MODEL}), or the DUPLEXEVAL_* / OPENAI_* fallbacks."
        )
    return env


def _max_judge_attempts() -> int:
    try:
        return max(1, int(os.environ.get("FDB_JUDGE_MAX_ATTEMPTS", "5")))
    except ValueError:
        return 5


def judge_behaviour(payload: str) -> List[str]:
    """Return the judge's C-axis tags for one sample, as in the official eval_behavior.py."""
    import requests

    env = _judge_env()
    attempts = _max_judge_attempts()
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                f"{env['base_url'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {env['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": env["model"],
                    "messages": [
                        {"role": "system", "content": BEHAVIOR_INSTRUCTION},
                        {"role": "user", "content": payload},
                    ],
                    "seed": attempt + 1,
                },
                timeout=180,
            )
            if response.status_code in _FATAL_HTTP_STATUSES:
                raise JudgeConfigError(f"Full-Duplex-Bench judge endpoint rejected the request (HTTP {response.status_code}): {response.text[:200]}")
            response.raise_for_status()
            return _extract_behaviour(response.json()["choices"][0]["message"]["content"])
        except JudgeConfigError:
            raise
        except Exception as exc:
            last_error = exc
            eval_logger.warning(f"FDB judge attempt {attempt + 1}/{attempts} failed: {str(exc)[:200]}")
            if attempt + 1 < attempts:
                time.sleep(JUDGE_RETRY_SECONDS)
    raise RuntimeError(f"Full-Duplex-Bench judge failed after {attempts} attempts: {last_error}")


def _extract_behaviour(prediction: str) -> List[str]:
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


def _parse_sessions(raw: str) -> Tuple[List[List[Dict[str, Any]]], Optional[str]]:
    """Return the clean and overlapped chunk logs, and the generation error if any."""
    try:
        payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
    except json.JSONDecodeError as exc:
        return [[], []], f"model output is not JSON: {exc}"
    if not isinstance(payload, dict):
        return [[], []], "model output is empty or not a duplex session log"
    if payload.get("error"):
        return [[], []], str(payload["error"])
    sessions = list(payload.get("sessions", []))
    while len(sessions) < 2:
        sessions.append([])
    return sessions[:2], None


def _transcript(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The official ASR file format: ``{"text": ..., "chunks": [word, ...]}``."""
    return {"text": " ".join(word["text"] for word in words), "chunks": words}


def build_judge_payload(doc: dict, clean_chunks: List[Dict[str, Any]], noisy_chunks: List[Dict[str, Any]]) -> str:
    """The user message of the official judge call, character for character."""
    input_clean = _compact(json.loads(doc["input_clean"]))
    input_noisy = _compact(json.loads(doc["input_noisy"]))
    output_clean = _compact(_transcript(chunks_to_word_transcript(clean_chunks)))
    output_noisy = _compact(_transcript(chunks_to_word_transcript(noisy_chunks)))
    return f"""
        {{
            "input_clean": {input_clean},
            "input_noisy": {input_noisy},
            "output_clean": {output_clean},
            "output_noisy": {output_noisy}
        }}
        """


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Official report: the share of each C-axis tag (eval_behavior.py stats_by_axis).
LABEL_METRICS = {
    "C_RESPOND": "fdb_c_respond",
    "C_RESUME": "fdb_c_resume",
    "C_UNCERTAIN_HANDLING": "fdb_c_uncertain_handling",
    "C_UNKNOWN": "fdb_c_unknown",
}


def _unscored(reason: str, doc: dict) -> Dict[str, Optional[float]]:
    """A sample that could not be scored: no label share, counted in fdb_scoring_failed."""
    eval_logger.error(f"Full-Duplex-Bench sample {doc.get('sample_id')} not scored: {reason}")
    return {"fdb_expected_behavior": None, **{metric: None for metric in LABEL_METRICS.values()}, "fdb_scoring_failed": 1.0}


def fdb_process_results(doc: dict, results: List[str]) -> Dict[str, Optional[float]]:
    # Missing credentials stop the run here, before any sample is scored.
    _judge_env()
    (clean_chunks, noisy_chunks), error = _parse_sessions(results[0] if results else "")
    if error:
        return _unscored(f"generation failed ({error[:200]})", doc)

    try:
        labels = judge_behaviour(build_judge_payload(doc, clean_chunks, noisy_chunks))
    except JudgeConfigError:
        raise
    except Exception as exc:
        return _unscored(f"judge failed ({str(exc)[:200]})", doc)
    if not labels:
        return _unscored("judge returned no label", doc)

    # The instruction asks for exactly one C_* tag per sample; the official
    # stats_by_axis counts every tag, which is the same for one tag.
    label = labels[0]
    scores: Dict[str, Optional[float]] = {"fdb_expected_behavior": float(label in doc["accepted_labels"])}
    scores.update({metric: float(label == tag) for tag, metric in LABEL_METRICS.items()})
    scores["fdb_scoring_failed"] = 0.0
    return scores


def fdb_aggregate_mean(results: List[Optional[float]]) -> float:
    """Mean over the scored samples; unscored samples (None) are left out."""
    values = [float(value) for value in results if value is not None]
    return sum(values) / len(values) if values else 0.0
