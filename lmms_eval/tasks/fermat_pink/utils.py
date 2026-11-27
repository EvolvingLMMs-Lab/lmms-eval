"""FERMAT-PINK: over-correction-aware evaluation of handwritten math OCR.

Seong et al., "When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the
Evaluation of Multi-line Handwritten Math OCR", EMNLP 2026. https://github.com/9115jin/PINK

The VLM transcribes a handwritten student solution that contains an error. An LLM judge
grades the transcription and the oracle transcription (what the student actually wrote)
with a 5-component rubric. Components on which the transcription scores higher than the
oracle are over-corrections and are penalized (paper Eq. 2-4).

Scoring uses the `pink-eval` package (`pip install pink-eval`). Judge settings:
  PINK_JUDGE           judge preset: "gpt-5" (default, needs OPENAI_API_KEY) or "qwen3-next-80b"
  PINK_JUDGE_BASE_URL  OpenAI-compatible endpoint of the judge (e.g. a local vLLM server)
  PINK_JUDGE_MODEL     judge model name (overrides the preset)
  PINK_JUDGE_WORKERS   parallel judge requests (default 16)
  PINK_ORACLE_GRADES   oracle grade file for the chosen judge. By default the grades bundled
                       with pink-eval are used when available (e.g. the paper's GPT-5 grades);
                       otherwise the oracle transcriptions are graded on the fly.
"""

import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor

from loguru import logger as eval_logger

# OCR prompt of the paper (Fig. 18).
OCR_SYSTEM_PROMPT = (
    "You are a math assistant specializing in extracting mathematical question and answer content from "
    "handwritten images of math problems by middle or high school students. Your task is to analyze the given "
    "Image, which contains the handwritten math Question-Answer pair, and convert it into LaTeX format. Follow "
    "the specific instructions provided carefully and output your response strictly in the requested format. "
    "Ensure that your evaluation is precise, accurate, and based solely on the information visible in the image."
)
OCR_USER_PROMPT = (
    "The image provided contains a handwritten math problem with both a Question and an Answer at a middle or "
    "high school level. Your task is to explicitly perform OCR on the handwritten text and extract the content "
    "in LaTeX format. Return only the extracted content, formatted in LaTeX, exactly as it appears in the image, "
    "in the **Question:** and **Answer:** fields.\n\n"
    "Ensure that no extra information is added that is not in the image.\n\n"
    "Please return the LaTeX output as follows:\n"
    "**Question:**<Extracted Question text in LaTeX>\n"
    "**Answer:**<Extracted Answer text in LaTeX>"
)

# lmms-eval imports this file once per `!function` reference, so a plain module-level dict would
# not be shared between the three aggregations. Keep the cache in sys.modules instead, so the
# transcriptions are judged once per run.
_CACHE_MODULE = sys.modules.setdefault("_lmms_eval_fermat_pink_cache", types.ModuleType("_lmms_eval_fermat_pink_cache"))
if not hasattr(_CACHE_MODULE, "results"):
    _CACHE_MODULE.results = {}
_RESULT_CACHE = _CACHE_MODULE.results


def _pink():
    try:
        import pink  # noqa: F401
        from pink import data, judge, ocr, score
    except ImportError as e:
        raise ImportError("fermat_pink is scored with PINK: `pip install 'pink-eval>=1.1'`") from e
    return data, judge, ocr, score


def fermat_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def fermat_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return OCR_USER_PROMPT


def fermat_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return [
        {"role": "system", "content": [{"type": "text", "text": OCR_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "url": fermat_doc_to_visual(doc)[0]}, {"type": "text", "text": fermat_doc_to_text(doc)}]},
    ]


def _oracle_text(doc):
    data, _, _, _ = _pink()
    return data.oracle_transcription(doc["pert_a"])


def fermat_doc_to_target(doc):
    return _oracle_text(doc)


def fermat_process_results(doc, results):
    item = {
        "sample_id": doc["new_custom_id"],
        "problem": str(doc["orig_q"]).replace("\r\n", "\n"),
        "oracle": _oracle_text(doc),
        "prediction": results[0] if results else "",
    }
    return {"pink": item, "over_correction_rate": item, "raw_score_ratio": item}


def _judge_config():
    _, judge, _, _ = _pink()
    preset = os.getenv("PINK_JUDGE", "gpt-5")
    return preset, judge.judge_from_preset(
        preset,
        base_url=os.getenv("PINK_JUDGE_BASE_URL") or None,
        model=os.getenv("PINK_JUDGE_MODEL") or None,
        api_key=os.getenv("PINK_JUDGE_API_KEY") or None,
    )


def _grade_all(items, texts, cfg, client, workers):
    _, judge, _, _ = _pink()

    def one(pair):
        item, text = pair
        return judge.grade(client, cfg, item["problem"], text, item["sample_id"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, zip(items, texts)))


def _score(items):
    key = tuple((it["sample_id"], it["prediction"]) for it in items)
    if key in _RESULT_CACHE:
        return _RESULT_CACHE[key]

    import pandas as pd

    _, judge, ocr, score = _pink()
    from pink.rubric import SCORE_COLUMNS

    preset, cfg = _judge_config()
    client = judge.make_client(cfg)
    workers = int(os.getenv("PINK_JUDGE_WORKERS", "16"))

    # Model grades. An empty transcription is a failed transcription and scores 0.
    texts = [ocr.clean_transcription(it["prediction"]) for it in items]
    rows = _grade_all(items, texts, cfg, client, workers)
    for row, text in zip(rows, texts):
        if not text.strip():
            row.update({"success": True, **{c: 0 for c in SCORE_COLUMNS}})
    model = pd.DataFrame([r for r in rows if r.get("success")])
    model = model.set_index("sample_id")[SCORE_COLUMNS].astype(float) if len(model) else pd.DataFrame(columns=SCORE_COLUMNS)

    # Oracle grades: user file > bundled grades for this judge > grade the oracle now.
    oracle_spec = os.getenv("PINK_ORACLE_GRADES") or (f"fermat:{preset}" if f"fermat:{preset}" in score.PACKAGED_ORACLES and not os.getenv("PINK_JUDGE_MODEL") else None)
    if oracle_spec:
        oracle = score.load_grades(oracle_spec)
        source = oracle_spec
    else:
        orows = _grade_all(items, [it["oracle"] for it in items], cfg, client, workers)
        oracle = pd.DataFrame([r for r in orows if r.get("success")]).set_index("sample_id")[SCORE_COLUMNS].astype(float)
        source = f"graded now with {cfg.model}"

    result = score.compute_pink(model, oracle)
    n_fail = sum(1 for r in rows if not r.get("success"))
    eval_logger.info(f"[fermat_pink] judge={cfg.model} oracle={source} n={result.n} judge_failures={n_fail} PINK={result.pink:.4f} over_correction_rate={result.over_correction_rate:.4f} raw={result.raw_ratio:.4f}")
    _RESULT_CACHE[key] = result
    return result


def fermat_pink_aggregate(results):
    return _score(results).pink


def fermat_over_correction_aggregate(results):
    return _score(results).over_correction_rate


def fermat_raw_ratio_aggregate(results):
    return _score(results).raw_ratio
