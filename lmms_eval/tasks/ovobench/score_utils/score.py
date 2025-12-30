# code from https://github.com/JoeLeelyf/OVO-Bench/blob/main/utils/OVOBenchScore.py
def calculate_score_backward_realtime(results):
    def get_score(response, gt):
        if response == None:
            return 0
        return int(gt in response)

    # Calculate Score for Every Result
    for i in range(len(results)):
        results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])

    scores = {}
    for i in range(len(results)):
        if not results[i]["task"] in scores.keys():
            scores[results[i]["task"]] = [results[i]["score"]]
        else:
            scores[results[i]["task"]].append(results[i]["score"])
    return results, scores


def calculate_score_forward(results):
    def get_score_REC(response, gt):
        if response == None:
            return 0
        import re

        response = re.findall(r"\d+", response)
        response = "".join(response)
        return response == str(gt)

    def get_score_SSR_CRR(response, gt):
        if response == None:
            return 0
        return int(gt in response)

    scores = {}
    tasks = list(set([result["task"] for result in results]))
    for task in tasks:
        scores[task] = []
    for i, result in enumerate(results):
        # Calculate score for REC
        if result["task"] == "REC":
            for j, test_info_ in enumerate(result["test_info"]):
                if test_info_["response"] is None:
                    continue
                scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
        # Calculate score for SSR
        if result["task"] == "SSR":
            for j, test_info_ in enumerate(result["test_info"]):
                if test_info_["response"] is None:
                    continue
                if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                    scores["SSR"].append(1)
                    continue
                gt = "No" if test_info_["type"] == 0 else "Yes"
                scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
        # Calculate score for CRR
        if result["task"] == "CRR":
            for j, test_info_ in enumerate(result["test_info"]):
                if test_info_["response"] is None:
                    continue
                if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                    scores["CRR"].append(1)
                    continue
                gt = "No" if test_info_["type"] == 0 else "Yes"
                scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
    return results, scores
