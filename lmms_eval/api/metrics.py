# the code is adapted from https://github.com/EleutherAI/lm-evaluation-harness
import collections
import math
import random
import re
import string
from collections.abc import Iterable
from typing import Any, List

import numpy as np
import sacrebleu

from lmms_eval.api.registry import register_aggregation, register_metric


# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("median")
def median(arr):
    return arr[len(arr) // 2]


# Certain metrics must be calculated across all documents in a benchmark.
# We use them as aggregation metrics, paired with no-op passthrough metric fns.
@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_aggregation("f1")
def f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)

    return np.max(fscore)


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


@register_aggregation("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("chrf")
def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_aggregation("brier_score")
def brier_score(items):  # This is a passthrough function
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape

    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score",
)
def brier_score_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


###


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="bits_per_byte",
)
def bits_per_byte_fn(items):  # This is a passthrough function
    return items


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


@register_metric(
    metric="anls",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def anls(
    references,
    predictions,
    thresh_hold=0.5,
):
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    # Unwrap predictions if it's a nested list
    pred = predictions[0] if isinstance(predictions[0], str) else predictions[0][0]

    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return {"anls": question_result}


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=[
        "loglikelihood",
        "multiple_choice",
        "generate_until",
        "generate_until_multi_round",
    ],
    aggregation="bypass",
)
def bypass(items):
    return items


@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="f1",
)
def f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type=["generate_until", "generate_until_multi_round"],
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type=["generate_until", "generate_until_multi_round"],
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=True,
    output_type=["generate_until", "generate_until_multi_round"],
    aggregation="ter",
)
def ter_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# stderr stuff


class _bootstrap_internal:
    def __init__(self, f, n) -> None:
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print("bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def bootstrap_chair_metric(metric_fn, xs, iters):
    "for non multiprocessing for CHAIR"
    print(f"bootstrapping for stddev: {metric_fn.__name__}")
    res = []
    from tqdm import tqdm

    for _ in tqdm(range(iters), desc="Bootstrap"):
        bootstrap_sample = random.choices(xs, k=len(xs))
        metric_value = metric_fn(bootstrap_sample)
        res.append(metric_value)

    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters: int):
    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
    ]

    # Optional imports for tasks with extra dependencies (spacy, etc.)
    try:
        from lmms_eval.tasks.amber_g.utils import (
            amber_g_aggregate_chair,
            amber_g_aggregate_cog,
            amber_g_aggregate_cover,
            amber_g_aggregate_hal,
        )

        bootstrappable.extend(
            [
                amber_g_aggregate_chair,
                amber_g_aggregate_cover,
                amber_g_aggregate_hal,
                amber_g_aggregate_cog,
            ]
        )
    except ImportError:
        pass

    try:
        from lmms_eval.tasks.coco_cap_chair.utils import (
            coco_cap_chair_aggregate_results_chair_i,
            coco_cap_chair_aggregate_results_chair_s,
            coco_cap_chair_aggregate_results_recall,
        )

        bootstrappable.extend(
            [
                coco_cap_chair_aggregate_results_chair_i,
                coco_cap_chair_aggregate_results_chair_s,
                coco_cap_chair_aggregate_results_recall,
            ]
        )
    except ImportError:
        pass

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    if hasattr(metric, "__name__"):
        if "coco_cap_chair" in metric.__name__:
            return lambda x: bootstrap_chair_metric(metric, x, iters=bootstrap_iters)
        if "amber_g" in metric.__name__ or "amber_" in metric.__name__:
            return lambda x: bootstrap_chair_metric(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
    assert metrics is not None, "Need to pass a list of each subtask's metric for this stderr aggregation"
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
        curr_score = ((curr_score * curr_size) + (score * size)) / (curr_size + size)  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (curr_size + size - 1) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (curr_score - score) ** 2

    return np.sqrt(variance)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)


def expected_accuracy(sample_scores: List[List[float]]) -> float:
    """
    Calculate Expected Accuracy (EA) - average accuracy over k samples.

    Args:
        sample_scores: List of lists, where each inner list contains k scores
                       for a single question (e.g., [[0,1,1], [1,1,0], ...])

    Returns:
        EA: mean of all individual sample scores
    """
    if not sample_scores:
        return float("nan")
    all_scores = [s for scores in sample_scores for s in scores]
    return sum(all_scores) / len(all_scores) if all_scores else float("nan")


def consensus_accuracy(sample_scores: List[List[float]]) -> float:
    """
    Calculate Consensus Accuracy (CA) via majority voting.

    For each question, take the majority vote across k samples.
    CA = fraction of questions where majority vote is correct.

    Args:
        sample_scores: List of lists of 0/1 scores per question

    Returns:
        CA: accuracy after majority voting
    """
    if not sample_scores:
        return float("nan")
    correct = 0
    for scores in sample_scores:
        if not scores:
            continue
        # Majority vote: correct if more than half are 1
        if sum(scores) > len(scores) / 2:
            correct += 1
    return correct / len(sample_scores) if sample_scores else float("nan")


def internal_variance(sample_scores: List[List[float]]) -> float:
    """
    Calculate Internal Variance (IV) - average variance within each question.

    Lower IV indicates more consistent/stable model behavior.

    Args:
        sample_scores: List of lists of scores per question

    Returns:
        IV: mean of per-question variances
    """
    if not sample_scores:
        return float("nan")
    variances = []
    for scores in sample_scores:
        if len(scores) < 2:
            continue
        mean_s = sum(scores) / len(scores)
        var = sum((s - mean_s) ** 2 for s in scores) / (len(scores) - 1)
        variances.append(var)
    return sum(variances) / len(variances) if variances else float("nan")


def consistency_rate(sample_scores: List[List[float]]) -> float:
    """
    Calculate Consistency Rate (CR) - fraction of questions with consistent answers.

    A question is consistent if all k samples give the same answer.

    Args:
        sample_scores: List of lists of 0/1 scores per question

    Returns:
        CR: fraction of questions with all-same answers
    """
    if not sample_scores:
        return float("nan")
    consistent = 0
    for scores in sample_scores:
        if not scores:
            continue
        # Consistent if all scores are the same (all 0 or all 1)
        if len(set(scores)) == 1:
            consistent += 1
    return consistent / len(sample_scores) if sample_scores else float("nan")


def clustered_stderr(scores: List[float], cluster_ids: List[Any]) -> float:
    """
    Calculate clustered standard error for non-independent samples.

    When multiple questions share the same context (e.g., same image/video),
    they are not independent. This implements Equation 4 from:
    "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"
    (https://arxiv.org/abs/2411.00640)

    SE_clustered = sqrt(SE_CLT^2 + (1/n^2) * sum_c sum_i sum_{j!=i} (s_ic - s_bar)(s_jc - s_bar))

    Args:
        scores: List of individual scores (e.g., 0/1 for correctness)
        cluster_ids: List of cluster identifiers (e.g., video_id, image_id)

    Returns:
        Clustered standard error, or NaN if insufficient data
    """
    n = len(scores)
    if n < 2:
        return float("nan")

    if len(scores) != len(cluster_ids):
        raise ValueError("scores and cluster_ids must have the same length")

    # Global mean
    s_bar = sum(scores) / n

    # SE_CLT^2 = Var(scores) / n = (1/(n-1)) * sum((s_i - s_bar)^2) / n
    var_scores = sum((s - s_bar) ** 2 for s in scores) / (n - 1)
    se_clt_squared = var_scores / n

    # Group scores by cluster with their indices
    cluster_to_scores = collections.defaultdict(list)
    for i, (score, cid) in enumerate(zip(scores, cluster_ids)):
        cluster_to_scores[cid].append(score)

    # Calculate within-cluster cross-terms: sum_c sum_i sum_{j!=i} (s_ic - s_bar)(s_jc - s_bar)
    cross_term = 0.0
    for cid, cluster_scores in cluster_to_scores.items():
        # For each cluster, compute sum of (s_i - s_bar)(s_j - s_bar) for i != j
        deviations = [s - s_bar for s in cluster_scores]
        cluster_sum = sum(deviations)
        # sum_{i!=j} d_i * d_j = (sum d_i)^2 - sum(d_i^2)
        sum_of_squares = sum(d * d for d in deviations)
        cross_term += cluster_sum * cluster_sum - sum_of_squares

    cross_term /= n * n

    # SE_clustered = sqrt(SE_CLT^2 + cross_term)
    return math.sqrt(se_clt_squared + cross_term)


def paired_ttest(current_scores: List[float], baseline_scores: List[float]) -> dict:
    """
    Perform paired t-test comparing current model scores against baseline.

    This implements a paired-differences test for model comparison, computing:
    - Mean difference (current - baseline)
    - Standard error of the difference
    - 95% confidence interval
    - t-statistic and p-value

    Args:
        current_scores: List of scores from the current model (per sample)
        baseline_scores: List of scores from the baseline model (per sample)

    Returns:
        dict with keys: mean_diff, se_diff, ci_lower, ci_upper, t_stat, p_value, n
    """
    from scipy import stats

    if len(current_scores) != len(baseline_scores):
        raise ValueError(f"Score lists must have same length: current={len(current_scores)}, baseline={len(baseline_scores)}")

    n = len(current_scores)
    if n < 2:
        return {
            "mean_diff": float("nan"),
            "se_diff": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }

    diffs = [c - b for c, b in zip(current_scores, baseline_scores)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    se_diff = math.sqrt(var_diff / n)

    if se_diff == 0:
        t_stat = float("inf") if mean_diff > 0 else float("-inf") if mean_diff < 0 else 0.0
        p_value = 0.0 if mean_diff != 0 else 1.0
    else:
        t_stat = mean_diff / se_diff
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value": p_value,
        "n": n,
    }


def power_analysis(
    effect_size: float,
    std_a: float = None,
    std_b: float = None,
    alpha: float = 0.05,
    power: float = 0.80,
    correlation: float = 0.5,
    current_n: int = None,
) -> dict:
    """
    Calculate minimum sample size for paired t-test power analysis.

    For paired samples, the effective variance is:
        Var(X - Y) = Var(X) + Var(Y) - 2*Cov(X,Y)
                   = std_a^2 + std_b^2 - 2*rho*std_a*std_b

    Formula (from Miller 2024, "Adding Error Bars to Evals"):
        n = ((z_alpha + z_beta) / d)^2
    where d = effect_size / std_diff is the standardized effect size.

    Note: std_a and std_b should ideally be estimated from previous
    evaluation data rather than using the default values.
    See: https://arxiv.org/abs/2411.00640 Section 5 for details.

    Args:
        effect_size: Minimum detectable difference (e.g., 0.03 for 3%)
        std_a: Std deviation of model A scores (estimate from previous eval)
        std_b: Std deviation of model B scores (estimate from previous eval)
               If only std_a provided, assumes std_b = std_a
               If neither provided, defaults to 0.5 (binary 0/1 approximation)
        alpha: Significance level (default 0.05)
        power: Desired statistical power (default 0.80)
        correlation: Expected correlation between paired samples (default 0.5)
        current_n: If provided, also compute the power for this sample size

    Returns:
        Dictionary with min_n, current_power (if current_n provided), and other details
    """
    from scipy import stats

    # Handle std defaults: if neither provided, use 0.5; if only std_a, assume equal
    if std_a is None and std_b is None:
        std_a = std_b = 0.5  # Default for binary (0/1) scores
    elif std_a is not None and std_b is None:
        std_b = std_a  # Assume equal variance if only one provided
    elif std_a is None and std_b is not None:
        std_a = std_b

    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)

    # General formula: Var(X-Y) = Var(X) + Var(Y) - 2*Cov(X,Y)
    # where Cov(X,Y) = rho * std_a * std_b
    var_diff = std_a**2 + std_b**2 - 2 * correlation * std_a * std_b
    std_diff = math.sqrt(var_diff)
    d = effect_size / std_diff
    min_n = math.ceil(((z_alpha + z_beta) / d) ** 2)

    result = {
        "min_n": min_n,
        "effect_size": effect_size,
        "std_a": std_a,
        "std_b": std_b,
        "alpha": alpha,
        "power": power,
        "correlation": correlation,
    }

    if current_n is not None:
        achieved_z_beta = d * math.sqrt(current_n) - z_alpha
        achieved_power = stats.norm.cdf(achieved_z_beta)
        result["current_n"] = current_n
        result["current_power"] = round(achieved_power, 4)
        mde = (z_alpha + z_beta) * std_diff / math.sqrt(current_n)
        result["min_detectable_effect"] = round(mde, 4)

    return result
