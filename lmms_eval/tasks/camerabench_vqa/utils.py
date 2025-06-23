import os
import re

dir_name = os.path.dirname(os.path.abspath(__file__))

SUFFIX_FOR_VQA = {"yes_no": "Please answer Yes or No.", "multiple_choice": "Please output the letter corresponding to the correct option."}



def get_scores(scores):
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0
        - "q0_i1" means question_0 on image_1
        - "q1_i0" means question_1 on image_0
        - "q1_i1" means question_1 on image_1

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'Acc': Average binary VQA acc
            - 'Q_Acc': Average question acc
            - 'I_Acc': Average image acc
            - 'G_Acc': Average group acc
    """
    Q_Acc = 0.0
    I_Acc = 0.0
    Acc = 0.0
    G_Acc = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct

    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group_score(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1

        return group_correct

    if isinstance(scores, dict):
        for _, result in scores.items():
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group_score(result)
    else:
        for result in scores:
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group_score(result)

    results = {"Q_Acc": Q_Acc / float(num_samples * 2), "I_Acc": I_Acc / float(num_samples * 2), "Acc": Acc / float(num_samples * 4), "G_Acc": G_Acc / num_samples}

    return results


def extract_answer(output_string, task_type="yes_no"):
    """
    Extracts the answer from the output string based on the task type.

    Parameters:
    output_string (str): The output string.
    task_type (str): The type of task. Must be "yes_no" as CameraBench does not have "multiple_choice" questions.

    Returns:
    int:
        1 if "yes" or "A"
        0 if "no" or "B"
        -1 if no relevant answer is found.
        Raises a ValueError if an unsupported task_type is provided.
    """

    def find_word_position(string, word):
        pattern = r"\b" + re.escape(word) + r"\b"
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1

    if task_type != "yes_no":
        raise ValueError("Task type not supported. Must be 'yes_no'. CameraBench VQA only have 'yes_no' questions.")

    # if task_type == "yes_no":
    position_yes_and_a = find_word_position(output_string, "yes")
    position_no_and_b = find_word_position(output_string, "no")
    # elif task_type == "multiple_choice":
    #     position_yes_and_a = find_word_position(output_string, "A")
    #     position_no_and_b = find_word_position(output_string, "B")

    if position_yes_and_a == -1 and position_no_and_b == -1:
        print(f"No answer found in the output string: {output_string}.")
        return -1
    elif position_yes_and_a != -1 and position_no_and_b != -1:
        return 1 if position_yes_and_a < position_no_and_b else 0
    else:
        return 0 if position_yes_and_a == -1 else 1


def cambench_doc_to_visual(doc):
    try:
        default_path = os.path.join(os.getenv('HOME'), '.cache/huggingface')
        load_path = os.path.expanduser(os.path.join(
            os.getenv("HF_HOME", default_path),
            'camerabench_vqa/datasets--chancharikm--camerabench_vqa_lmms_eval/snapshots'
        ))

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Dataset path not found: {load_path}")

        snapshots = os.listdir(load_path)
        if not snapshots:
            raise FileNotFoundError(f"No snapshots found in: {load_path}")

        snapshot_path = os.path.join(load_path, snapshots[0])
        video_path = os.path.join(snapshot_path, doc["Video"])

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        return [video_path]
    except Exception as e:
        eval_logger.error(f"Error constructing video path: {e}")
        raise


def cambench_doc_to_text(doc):
    question = doc["Question"]
    question = question + " " + SUFFIX_FOR_VQA["yes_no"]
    # if doc["Question_Type"] == "yes_no":
    #     question = question + " " + SUFFIX_FOR_VQA["yes_no"]
    # elif doc["Question_Type"] == "multiple_choice":
    #     question = question + " " + SUFFIX_FOR_VQA["multiple_choice"]
    return question


def cambench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    # type = doc["Question_Type"]
    gt_ans = extract_answer(pred, task_type="yes_no")
    return {
        "cambench_G_ACC": {"id": doc["Index"], "score": gt_ans},
        "cambench_Q_ACC": {"id": doc["Index"], "score": gt_ans},
        "cambench_I_ACC": {"id": doc["Index"], "score": gt_ans},
        "cambench_Acc": {"id": doc["Index"], "score": gt_ans},
    }


def cambench_aggregate_results_G_ACC(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    assert len(results) == 1900 * 4
    answers = {}
    number_answered_samples = len(results) // 4
    for i in range(number_answered_samples):
        assert int(results[i * 4]["id"]) == i * 4
        assert int(results[i * 4 + 1]["id"]) == i * 4 + 1
        assert int(results[i * 4 + 2]["id"]) == i * 4 + 2
        assert int(results[i * 4 + 3]["id"]) == i * 4 + 3
        answers[i] = {"q0_i0": results[i * 4]["score"], "q0_i1": results[i * 4 + 1]["score"], "q1_i0": results[i * 4 + 2]["score"], "q1_i1": results[i * 4 + 3]["score"]}

    scores = get_scores(answers)

    # eval_logger.info(f"G_Acc: {scores["G_Acc"]:.2f}")

    return scores["G_Acc"]


def cambench_aggregate_results_Q_ACC(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    assert len(results) == 1900 * 4
    answers = {}
    number_answered_samples = len(results) // 4
    for i in range(number_answered_samples):
        assert int(results[i * 4]["id"]) == i * 4
        assert int(results[i * 4 + 1]["id"]) == i * 4 + 1
        assert int(results[i * 4 + 2]["id"]) == i * 4 + 2
        assert int(results[i * 4 + 3]["id"]) == i * 4 + 3
        answers[i] = {"q0_i0": results[i * 4]["score"], "q0_i1": results[i * 4 + 1]["score"], "q1_i0": results[i * 4 + 2]["score"], "q1_i1": results[i * 4 + 3]["score"]}

    scores = get_scores(answers)

    # eval_logger.info(f"Q_Acc: {scores["Q_Acc"]:.2f}")

    return scores["Q_Acc"]


def cambench_aggregate_results_I_ACC(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    assert len(results) == 1900 * 4
    answers = {}
    number_answered_samples = len(results) // 4
    for i in range(number_answered_samples):
        assert int(results[i * 4]["id"]) == i * 4
        assert int(results[i * 4 + 1]["id"]) == i * 4 + 1
        assert int(results[i * 4 + 2]["id"]) == i * 4 + 2
        assert int(results[i * 4 + 3]["id"]) == i * 4 + 3
        answers[i] = {"q0_i0": results[i * 4]["score"], "q0_i1": results[i * 4 + 1]["score"], "q1_i0": results[i * 4 + 2]["score"], "q1_i1": results[i * 4 + 3]["score"]}

    scores = get_scores(answers)

    # eval_logger.info(f"I_Acc: {scores["I_Acc"]:.2f}")

    return scores["I_Acc"]


def cambench_aggregate_results_ACC(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    assert len(results) == 1900 * 4
    answers = {}
    number_answered_samples = len(results) // 4
    for i in range(number_answered_samples):
        assert int(results[i * 4]["id"]) == i * 4
        assert int(results[i * 4 + 1]["id"]) == i * 4 + 1
        assert int(results[i * 4 + 2]["id"]) == i * 4 + 2
        assert int(results[i * 4 + 3]["id"]) == i * 4 + 3
        answers[i] = {"q0_i0": results[i * 4]["score"], "q0_i1": results[i * 4 + 1]["score"], "q1_i0": results[i * 4 + 2]["score"], "q1_i1": results[i * 4 + 3]["score"]}

    scores = get_scores(answers)

    # eval_logger.info(f"Acc: {scores["Acc"]:.2f}")

    return scores["Acc"]
