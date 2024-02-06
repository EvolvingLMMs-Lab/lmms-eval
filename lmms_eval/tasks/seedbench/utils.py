import json


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def seed_process_result(doc, result):
    pred = result[0].strip()
    answer = doc["answer"]
    data_type = doc["data_type"]

    return {f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}, f"seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"] == result["answer"]:
            total_correct += 1
        total_count += 1
    return total_correct / total_count


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score


def seed_doc_to_text_mc(doc):
    question = doc["question"]
    return f"{question} Answer :"


def seed_doc_to_choice(doc):
    return [doc["choice_a"], doc["choice_b"], doc["choice_c"], doc["choice_d"]]


def seed_doc_to_mc_target(doc):
    answer2choice = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
    return doc[answer2choice[doc["answer"]]]
