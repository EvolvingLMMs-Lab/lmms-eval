def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["prompt"]


def doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return []


def process_results(doc, results):
    result = results[0] if results else ""
    success = bool(result) and not result.startswith("[") and not result.startswith("ERROR")
    return {"generated": 1.0 if success else 0.0}


def count_generated(results):
    if not results:
        return 0.0
    return sum(results) / len(results)
