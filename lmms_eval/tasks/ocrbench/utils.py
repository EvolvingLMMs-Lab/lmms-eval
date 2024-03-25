import logging

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

logger = logging.getLogger("lmms-eval")

# Add the following functions to your existing utils.py file
OCRBench_score = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}


def ocrbench_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def ocrbench_doc_to_text(doc):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{question}"


def ocrbench_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answer"]
    dataset_name = doc["dataset"]

    score = 0
    if dataset_name == "HME100k":
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].strip().replace("\n", " ").replace(" ", "")
                predict = pred.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.strip().replace("\n", " ").replace(" ", "")
            predict = pred.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].lower().strip().replace("\n", " ")
                predict = pred.lower().strip().replace("\n", " ")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.lower().strip().replace("\n", " ")
            predict = pred.lower().strip().replace("\n", " ")
            if answer in predict:
                score = 1
    return {
        "ocrbench_accuracy": {"question_type": doc["question_type"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def ocrbench_aggregate_accuracy(results, args):
    for result in results:
        OCRBench_score[result["question_type"]] += result["score"]
    recognition_score = (
        OCRBench_score["Regular Text Recognition"]
        + OCRBench_score["Irregular Text Recognition"]
        + OCRBench_score["Artistic Text Recognition"]
        + OCRBench_score["Handwriting Recognition"]
        + OCRBench_score["Digit String Recognition"]
        + OCRBench_score["Non-Semantic Text Recognition"]
    )
    Final_score = recognition_score + OCRBench_score["Scene Text-centric VQA"] + OCRBench_score["Doc-oriented VQA"] + OCRBench_score["Key Information Extraction"] + OCRBench_score["Handwritten Mathematical Expression Recognition"]
    file_name = generate_submission_file("ocrbench_results.txt", args, subpath="results")
    with open(file_name, "w") as f:
        print("######################### OCRBench #############################", file=f)
        print(f"Text Recognition(Total 300): {recognition_score}", file=f)
        print("---------------- Details of Recognition Score ------------------", file=f)
        print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}", file=f)
        print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}", file=f)
        print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}", file=f)
        print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}", file=f)
        print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}", file=f)
        print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}", file=f)
        print("----------------------------------------------------------------", file=f)
        print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}", file=f)
        print("----------------------------------------------------------------", file=f)
        print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}", file=f)
        print("----------------------------------------------------------------", file=f)
        print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}", file=f)
        print("----------------------------------------------------------------")
        print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}", file=f)
        print("----------------------Final Score-------------------------------", file=f)
        print(f"Final Score(Total 1000): {Final_score}", file=f)
    logger.info(f"OCR Bench results saved to {file_name}")
    # return {"Final Score":Final_score,"Text Recognition":recognition_score,'Scene Text-centric VQA':OCRBench_score['Scene Text-centric VQA'],'Doc-oriented VQA':OCRBench_score['Doc-oriented VQA'],'Key Information Extraction':OCRBench_score['Key Information Extraction'],'Handwritten Mathematical Expression Recognition':OCRBench_score['Handwritten Mathematical Expression Recognition']}
    return Final_score
