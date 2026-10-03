import hashlib
import random
import re
from collections import defaultdict

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

MVH_TYPES = ("cross_instance", "cross_view")
SHUFFLE_SEED = 2025


def _expand_docs(dataset, mvh_type=None):
    """Expand each image-pair group into 2 MC + 4 binary QA documents."""
    if mvh_type is not None:
        dataset = dataset.filter(lambda x: x == mvh_type, input_columns=["mvh_type"])

    n = len(dataset)
    expanded = dataset.select([i for i in range(n) for _ in range(6)])
    return expanded.add_column("question_index", list(range(6)) * n)


def mvh_process_docs(dataset):
    return _expand_docs(dataset)


def mvh_process_docs_cross_view(dataset):
    return _expand_docs(dataset, "cross_view")


def mvh_process_docs_cross_instance(dataset):
    return _expand_docs(dataset, "cross_instance")


def _mc_choices(doc):
    """Shuffle all three MC options, including Neither, deterministically."""
    options = list(doc["mc_options"])
    qidx = doc["question_index"]

    key = f"{SHUFFLE_SEED}:{doc['group_id']}:{qidx}"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
    random.Random(seed).shuffle(options)

    letters = "ABC"
    gold = letters[options.index(doc["mc_answers"][qidx])]
    adversarial = letters[options.index(doc["mc_answers"][1 - qidx])]

    neither = next(x for x in letters if x not in (gold, adversarial))
    other = [x for x in letters if x != neither]
    options[letters.index(neither)] = f"Neither {other[0]} nor {other[1]}"
    return options, gold, adversarial


def mvh_doc_to_visual(doc):
    return [doc["view_1"].convert("RGB"), doc["view_2"].convert("RGB")]


def _build_question_prompt(doc):
    qidx = doc["question_index"]

    if qidx < 2:
        options, _, _ = _mc_choices(doc)
        choices = "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options))
        return f"Question:\n{doc['mc_questions'][qidx]}\n\nChoices:\n{choices}\n\nOnly one option is correct.\nPresent the answer strictly in the form X)."

    return doc["binary_questions"][qidx - 2] + "\nPlease answer this question with one word."


def mvh_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _build_question_prompt(doc)


def mvh_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": doc["view_1"].convert("RGB")},
                {"type": "image", "url": doc["view_2"].convert("RGB")},
                {"type": "text", "text": _build_question_prompt(doc)},
            ],
        }
    ]


def mvh_doc_to_target(doc):
    qidx = doc["question_index"]
    return _mc_choices(doc)[1] if qidx < 2 else doc["binary_answers"][qidx - 2]


def mvh_process_results(doc, results):
    response = results[0] if results else ""
    qidx = doc["question_index"]

    if qidx < 2:
        _, gold, adversarial = _mc_choices(doc)
        pred = extract_mcq_answer(response, choices=["A", "B", "C"])
        record = {
            "group_id": doc["group_id"],
            "mvh_type": doc["mvh_type"],
            "question_index": qidx,
            "correct": pred == gold,
            "adversarial": pred == adversarial,
        }
        return {name: record for name in ("mc_acc", "mc_pacc", "aer", "mvh_score")}

    matches = {x.lower() for x in re.findall(r"\b(yes|no)\b", response or "", flags=re.I)}
    pred = next(iter(matches)).capitalize() if len(matches) == 1 else ""
    gold = doc["binary_answers"][qidx - 2]
    record = {
        "group_id": doc["group_id"],
        "mvh_type": doc["mvh_type"],
        "question_index": qidx,
        "correct": pred == gold,
        "false_yes": pred == "Yes" and gold == "No",
    }
    return {name: record for name in ("binary_acc", "binary_pacc", "binary_qacc", "yer", "mvh_score")}


def mvh_process_results_full(doc, results):
    metrics = mvh_process_results(doc, results)
    prefix = doc["mvh_type"] + "_"
    return {**metrics, **{prefix + k: v for k, v in metrics.items()}}


def _groups(results):
    groups = defaultdict(dict)
    for r in results:
        groups[r["group_id"]][r["question_index"]] = r
    return groups


def mvh_aggregate_acc(results):
    if not results:
        return float("nan")
    return 100.0 * sum(r["correct"] for r in results) / len(results)


def _group_accuracy(results, index_sets):
    hits = []
    for group in _groups(results).values():
        for indices in index_sets:
            if all(i in group for i in indices):
                hits.append(all(group[i]["correct"] for i in indices))
    return 100.0 * sum(hits) / len(hits) if hits else float("nan")


def mvh_aggregate_mc_pacc(results):
    return _group_accuracy(results, [(0, 1)])


def mvh_aggregate_binary_pacc(results):
    return _group_accuracy(results, [(2, 3), (4, 5)])


def mvh_aggregate_binary_qacc(results):
    return _group_accuracy(results, [(2, 3, 4, 5)])


def mvh_aggregate_aer(results):
    errors = sum(not r["correct"] for r in results)
    return 100.0 * sum(r["adversarial"] for r in results) / errors if errors else 0.0


def mvh_aggregate_yer(results):
    errors = sum(not r["correct"] for r in results)
    return 100.0 * sum(r["false_yes"] for r in results) / errors if errors else 0.0


def _type_score(results):
    mc = [r for r in results if r["question_index"] < 2]
    binary = [r for r in results if r["question_index"] >= 2]
    return mvh_aggregate_acc(mc) + mvh_aggregate_mc_pacc(mc) + mvh_aggregate_acc(binary) + mvh_aggregate_binary_pacc(binary) + mvh_aggregate_binary_qacc(binary)


def mvh_aggregate_mvh_score(results):
    """MVH-Score for one MVH type: Acc+p-Acc+q-Acc+MC Acc+MC p-Acc."""
    return _type_score(results)


def mvh_aggregate_total_score(results):
    """Overall MVH-Score = cross-instance score + cross-view score."""
    return sum(_type_score([r for r in results if r["mvh_type"] == t]) for t in MVH_TYPES)
