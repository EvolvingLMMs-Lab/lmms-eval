import os

from lmms_eval.tasks.asr_wer_utils import (
    EvaluationTokenizer as SharedEvaluationTokenizer,
)
from lmms_eval.tasks.asr_wer_utils import compute_wer as shared_compute_wer
from lmms_eval.tasks.asr_wer_utils import remove_sp as shared_remove_sp
from lmms_eval.tasks.gigaspeech.whisper_normalizer.basic import BasicTextNormalizer
from lmms_eval.tasks.gigaspeech.whisper_normalizer.english import EnglishTextNormalizer

# ImportError: To support decoding audio files, please install 'librosa' and 'soundfile'.
english_normalizer = EnglishTextNormalizer()

basic_normalizer = BasicTextNormalizer()

dir_name = os.path.dirname(os.path.abspath(__file__))

SPECIAL_TOKENS = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTION>": "?",
    "<EXCLAMATION>": "!",
}


def gigaspeech_doc_to_audio(doc):
    return [doc["audio"]]


def gigaspeech_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}Please recognize the speech and only output the recognized content:{post_prompt}"


def gigaspeech_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""
    gt = doc["gt"]
    for token, replaced in SPECIAL_TOKENS.items():
        gt = gt.replace(token, replaced)
    data_dict = {"gt": gt, "pred": pred}

    return {"wer": data_dict}


def gigaspeech_xl_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""
    gt = doc["text"]
    for token, replaced in SPECIAL_TOKENS.items():
        gt = gt.replace(token, replaced)
    data_dict = {"gt": gt, "pred": pred}

    return {"wer": data_dict}


PUNCS = "!,.?;:"


def remove_sp(text):
    return shared_remove_sp(text, "", puncs=PUNCS)


EvaluationTokenizer = SharedEvaluationTokenizer


def compute_wer(refs, hyps):
    return shared_compute_wer(
        refs,
        hyps,
        "en",
        english_languages={"en"},
        english_normalizer=english_normalizer,
    )


def gigaspeech_wer(results, args):
    refs, hyps = [], []
    for result in results:
        gt = result["gt"]
        response = result["pred"]
        gt = remove_sp(gt)
        response = remove_sp(response)
        refs.append(gt)
        hyps.append(response)
    wer = compute_wer(refs, hyps)
    # print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")
    return wer * 100
