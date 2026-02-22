from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge


def get_requery_score(prediction, gt):
    score_dict = dict()

    # 计算BLEUBLEU分数
    smoothing_function = SmoothingFunction().method1  # * used to deal with non-overlap n-gram

    # calculate BLEU-1 score with smoothing function
    bleu_score = sentence_bleu([gt.split()], prediction.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(prediction, gt)[0]
    rouge_l_f1 = rouge_scores["rouge-l"]["f"]

    score_dict["bleu"] = bleu_score
    score_dict["rouge_l"] = rouge_l_f1
    score_dict["score"] = (bleu_score + rouge_l_f1) / 2

    return score_dict
