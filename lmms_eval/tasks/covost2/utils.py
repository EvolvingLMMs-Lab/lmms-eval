import os
import re
import unicodedata

from sacrebleu import corpus_bleu

from lmms_eval.tasks.gigaspeech.whisper_normalizer.basic import BasicTextNormalizer
from lmms_eval.tasks.gigaspeech.whisper_normalizer.english import EnglishTextNormalizer
from lmms_eval.tasks.librispeech.cn_tn import TextNorm

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

PUNCS = "!,.?;:"


def covost2_doc_to_audio(doc):
    return [doc["audio"]]


def covost2_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{post_prompt}"


def covost2_zh_en_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""
    gt = doc["gt"]
    data_dict = {"gt": gt, "pred": pred}
    return {"bleu": data_dict}


def covost2_en_zh_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""
    gt = doc["translation"]
    data_dict = {"gt": gt, "pred": pred}
    return {"bleu": data_dict}


def covost2_bleu(results, args):
    refs, hyps = [], []
    for result in results:
        gt = result["gt"]
        response = result["pred"]
        gt = remove_sp(gt)
        response = remove_sp(response)
        refs.append(gt)
        hyps.append(response)
    bleu = compute_bleu_zh(refs, hyps)
    return round(bleu, 5)


def covost2_bleu_en(results, args):
    refs, hyps = [], []
    for result in results:
        gt = result["gt"]
        response = result["pred"]
        gt = remove_sp(gt)
        response = remove_sp(response)
        refs.append(gt)
        hyps.append(response)
    bleu = compute_bleu_en(refs, hyps)
    return round(bleu, 5)


def remove_sp(text):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    return gt


def compute_bleu_en(refs, hyps):
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    refs = [tokenizer.tokenize(english_normalizer(ref)) for ref in refs]
    hyps = [tokenizer.tokenize(english_normalizer(hyp)) for hyp in hyps]
    bleu_score = corpus_bleu(hyps, [refs], tokenize="13a")
    return bleu_score.score


def compute_bleu_zh(refs, hyps):
    tokenizer = EvaluationTokenizer(
        tokenizer_type="zh",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    refs = [tokenizer.tokenize(chinese_normalizer(ref)) for ref in refs]
    hyps = [tokenizer.tokenize(chinese_normalizer(hyp)) for hyp in hyps]
    bleu_score = corpus_bleu(hyps, [refs], tokenize="zh")
    return bleu_score.score


class EvaluationTokenizer(object):
    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)

    def __init__(
        self,
        tokenizer_type: str = "13a",
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
        from sacrebleu.tokenizers.tokenizer_char import TokenizerChar
        from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
        from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
        from sacrebleu.tokenizers.tokenizer_none import NoneTokenizer
        from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

        TOKENIZERS = {
            "none": NoneTokenizer,
            "13a": Tokenizer13a,
            "intl": TokenizerV14International,
            "zh": TokenizerZh,
            "ja-mecab": TokenizerJaMecab,
            "char": TokenizerChar,
        }

        assert tokenizer_type in TOKENIZERS
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = TOKENIZERS[tokenizer_type]

    @classmethod
    def remove_punctuation(cls, sent: str):
        return cls.SPACE.join(t for t in sent.split(cls.SPACE) if not all(unicodedata.category(c)[0] == "P" for c in t))

    def tokenize(self, sent: str):
        tokenized = self.tokenizer()(sent)

        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)

        if self.character_tokenization:
            tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))

        if self.lowercase:
            tokenized = tokenized.lower()

        return tokenized
