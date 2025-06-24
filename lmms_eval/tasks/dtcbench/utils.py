def dtcbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def dtcbench_doc_to_text(doc):
    question = doc["question"]
    question += f"\nOptions: A: {doc['choice_a']}, B: {doc['choice_b']}, C: {doc['choice_c']}, D: {doc['choice_d']}"
    return f"{question}\n한 단어 또는 구를 사용하여 질문에 답하세요."


def dtcbench_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]

    return {"dtcbench_acc": {"pred": pred, "answer": answer, "question_id": doc["index"]}}


def dtcbench_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count
