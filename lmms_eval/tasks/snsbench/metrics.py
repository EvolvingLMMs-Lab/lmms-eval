"""
Modified from: https://github.com/HC-Guo/SNS-Bench-VL/blob/main/code/metrics.py
"""

import re
import string

import yaml
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

semantic_model = SentenceTransformer("BAAI/bge-large-en-v1.5")


def calculate_answer_accuracy(pred, gt):
    """
    计算答案完全匹配准确率
    """
    # letters_digits = string.digits + string.ascii_lowercase
    # pred = "".join([x for x in pred.lower() if x in letters_digits])
    # gt = "".join([x for x in gt.lower() if x in letters_digits])

    # 'C. Free shipping within the UK...' -> 'C'
    matches = re.findall(r"\b([A-Za-z])\.", pred)
    if matches:
        pred = matches[0]
    return 1 if pred in gt else 0


def calculate_three_level_accuracy(pred, gt):
    """
    计算三级准确率（词级别匹配）
    """
    ground_truth = gt.split(" ")
    model_output = pred

    # parse model output: "<text>A C D<text>" -> "A C D"
    match = re.search(r"([A-Z])\s+([A-Z])\s+([A-Z])", model_output)
    if match:
        model_output = match.group(0)

    model_output = model_output.split(" ")
    match_count = 0
    max_length = min(len(ground_truth), len(model_output))

    for i in range(max_length):
        if ground_truth[i] in model_output[i]:
            match_count += 1

    return match_count / 3 if max_length >= 3 else 0


def calculate_multiple_choice_f1(pred, gt):
    """
    计算多选题F1分数
    """
    # 处理标准答案
    reference_answers = gt.split(" ")
    reference_set = set(reference_answers)

    # 处理模型预测结果
    try:
        predicted_answers = pred.split(" ")
    except Exception as e:
        print(f"处理模型结果时出错: {e}")
        predicted_answers = []

    predicted_set = set(predicted_answers)

    # 计算TP、FP、FN
    true_positives = len(reference_set & predicted_set)
    false_positives = len(predicted_set - reference_set)
    false_negatives = len(reference_set - predicted_set)

    # 计算精确率、召回率和F1分数
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def calculate_semantic_similarity(pred, gt):
    """
    计算答案和模型输出的语义相似度分数
    """
    reference_text = gt
    predicted_text = pred

    # 生成向量表示
    embedding1 = semantic_model.encode(reference_text, normalize_embeddings=True)
    embedding2 = semantic_model.encode(predicted_text, normalize_embeddings=True)

    # 计算余弦相似度
    similarity_score = float(util.cos_sim(embedding1, embedding2).item())

    return similarity_score


def calculate_ocr_metrics(pred, gt, return_detailed=False):
    """
    计算答案质量的综合评估指标（ROUGE、BLEU和BGE语义相似度）
    """
    reference_text = gt
    predicted_text = pred

    # 1. 计算ROUGE分数
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, predicted_text)
    rouge_1 = rouge_scores["rouge1"].fmeasure
    rouge_l = rouge_scores["rougeL"].fmeasure
    avg_rouge = (rouge_1 + rouge_l) / 2

    # 2. 计算BLEU分数
    reference_tokens = [list(reference_text)]
    predicted_tokens = list(predicted_text)
    bleu_score = sentence_bleu(reference_tokens, predicted_tokens, smoothing_function=SmoothingFunction().method1)

    # 3. 计算BGE语义相似度
    embedding1 = semantic_model.encode(reference_text, normalize_embeddings=True)
    embedding2 = semantic_model.encode(predicted_text, normalize_embeddings=True)
    semantic_score = float(util.cos_sim(embedding1, embedding2).item())

    # 计算综合得分
    overall_score = (avg_rouge + bleu_score + semantic_score) / 3.0

    if return_detailed:
        return {"average_rouge": avg_rouge, "bleu_score": bleu_score, "semantic_similarity": semantic_score, "overall_quality": overall_score}
    else:
        return overall_score


if __name__ == "__main__":

    res = calculate_answer_accuracy("A. ", "A")
    res = calculate_three_level_accuracy("A C B", "A B C")
    res = calculate_multiple_choice_f1("A C B", "A B C")
    res = calculate_semantic_similarity("helly my friends", "hello buddies")
