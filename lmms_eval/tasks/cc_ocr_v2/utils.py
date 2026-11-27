"""CC-OCR V2 request and result adapters."""

from lmms_eval.tasks.cc_ocr_v2.scoring import score_sample


def doc_to_visual(doc):
    return list(doc["images"])


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"]


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    content = [{"type": "image", "url": page} for page in doc_to_visual(doc)]
    content.append({"type": "text", "text": doc_to_text(doc)})
    return [{"role": "user", "content": content}]


def process_results(doc, results):
    prediction = results[0]
    if prediction.startswith("[LMMS_EVAL_REQUEST_FAILED after ") or prediction == "[LMMS_EVAL_BUDGET_EXCEEDED]":
        raise RuntimeError(f"The backend did not return a model response for {doc['source_key']}")
    record = score_sample(doc, prediction)
    return {"cc_ocr_v2_average": record, f"cc_ocr_v2_{doc['track']}": record}
