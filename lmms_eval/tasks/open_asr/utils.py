import os

from lmms_eval.tasks.asr_wer_utils import (
    EvaluationTokenizer as SharedEvaluationTokenizer,
)
from lmms_eval.tasks.asr_wer_utils import compute_wer as shared_compute_wer
from lmms_eval.tasks.asr_wer_utils import remove_sp as shared_remove_sp
from lmms_eval.tasks.asr_wer_utils import (
    zhconv_convert,
)
from lmms_eval.tasks.librispeech.cn_tn import TextNorm
from lmms_eval.tasks.librispeech.whisper_normalizer.basic import BasicTextNormalizer
from lmms_eval.tasks.librispeech.whisper_normalizer.english import EnglishTextNormalizer

# ImportError: To support decoding audio files, please install 'librosa' and 'soundfile'.
english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)
basic_normalizer = BasicTextNormalizer()

dir_name = os.path.dirname(os.path.abspath(__file__))


def openasr_doc_to_audio(doc):
    # Accept multiple common audio field names across datasets
    for key in ["audio", "file", "path", "audio_path"]:
        if key in doc:
            return [doc[key]]
    raise KeyError(f"No audio field found. Tried ['audio','file','path','audio_path']. Available: {list(doc.keys())}")


def openasr_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}Please recognize the speech and only output the recognized content:{post_prompt}"


def openasr_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""

    # Normalize target field across datasets (e.g., 'text', 'transcript', 'gt')
    gt = openasr_doc_to_target(doc)

    data_dict = {"gt": gt, "pred": pred}

    return {"wer": data_dict}


def openasr_doc_to_target(doc):
    # Prefer 'text' (Open-ASR), fallback to 'transcript' or 'gt'
    for key in ["text", "transcript", "gt"]:
        if key in doc:
            return doc[key]
    raise KeyError(f"No target field found. Tried ['text','transcript','gt']. Available: {list(doc.keys())}")


PUNCS = "!,.?;:"


def remove_sp(text, language):
    return shared_remove_sp(text, language, puncs=PUNCS, no_space_languages={"zh"})


EvaluationTokenizer = SharedEvaluationTokenizer


def compute_wer(refs, hyps, language):
    return shared_compute_wer(
        refs,
        hyps,
        language,
        yue_languages={"yue"},
        english_languages={"en"},
        chinese_languages={"zh"},
        char_languages={"zh", "yue"},
        yue_converter=zhconv_convert,
        english_normalizer=english_normalizer,
        chinese_normalizer=chinese_normalizer,
        basic_normalizer=basic_normalizer,
    )


def openasr_wer(results, args):
    # lan = args["language"]
    refs, hyps = [], []
    lan = "en"
    # results_list = results_dict[source]
    for result in results:
        lan = "en"
        gt = result["gt"]
        response = result["pred"]
        gt = remove_sp(gt, lan)
        response = remove_sp(response, lan)
        refs.append(gt)
        hyps.append(response)
    wer = compute_wer(refs, hyps, lan)
    # print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")
    return wer * 100
