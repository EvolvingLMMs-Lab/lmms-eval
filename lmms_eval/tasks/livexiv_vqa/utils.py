import json
import re


def extract_answer(text):
    match = re.findall(r'(?<!^)[A-Z]', text)
    if match:
        return match[0]
    return None


def livexiv_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def livexiv_doc_to_text(doc, model_specific_kwargs=None):
    question = doc["question"]    
    question += "\n" + f"A. {doc['option_a']}\n"
    question += f"B. {doc['option_b']}\n"
    question += f"C. {doc['option_c']}\n"
    question += f"D. {doc['option_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def livexiv_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        if "answer" in pred.lower():
            pred = extract_answer(pred)
        else:
            pred = pred[0]
    answer = doc["gt"]

    return {f"livexiv_vqa": {"pred": pred, "answer": answer}}


def livexiv_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        try:
            if result["pred"].lower().strip() == result["answer"].lower().strip():
                total_correct += 1
        except Exception as e:
            print(e)

        total_count += 1
    return total_correct / total_count


def livexiv_aggregation_result_all(results):
    score = livexiv_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./livexiv_vqa_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for LiveXiv-VQA submission ...")

    return score


def livexiv_doc_to_text_mc(doc):
    question = doc["question"]
    return f"{question} Answer :"


def livexiv_doc_to_choice(doc):
    return [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]


def livexiv_doc_to_mc_target(doc):
    answer2choice = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}
    return doc[answer2choice[doc["answer"]]]
