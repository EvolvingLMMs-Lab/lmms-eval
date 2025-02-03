import os
import re
import unicodedata

import editdistance as ed  # TODO: new package
from sacrebleu import corpus_bleu

from lmms_eval.tasks.gigaspeech.whisper_normalizer.basic import BasicTextNormalizer
from lmms_eval.tasks.gigaspeech.whisper_normalizer.english import EnglishTextNormalizer

# ImportError: To support decoding audio files, please install 'librosa' and 'soundfile'.
english_normalizer = EnglishTextNormalizer()

basic_normalizer = BasicTextNormalizer()

dir_name = os.path.dirname(os.path.abspath(__file__))


def covost2_doc_to_audio(doc):
    return [doc["audio"]]


def covost2_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{post_prompt}"


def covost2_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""
    gt = doc["gt"]
    data_dict = {"gt": gt, "pred": pred}

    return {"bleu": data_dict}


PUNCS = "!,.?;:"


def remove_sp(text):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)  # Replace consecutive spaces in the text with a single space.
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    return gt


class EvaluationTokenizer(object):
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    Args:
        tokenizer_type (str): the type of sacreBLEU tokenizer to apply.
        lowercase (bool): lowercase the text.
        punctuation_removal (bool): remove punctuation (based on unicode
        category) from text.
        character_tokenization (bool): tokenize the text to characters.
    """

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)
    # ALL_TOKENIZER_TYPES = ChoiceEnum(["none", "13a", "intl", "zh", "ja-mecab"])

    def __init__(
        self,
        tokenizer_type: str = "13a",
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        # from sacrebleu.tokenizers import TOKENIZERS
        # from sacrebleu.tokenizers import tokenizer_none
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

        assert tokenizer_type in TOKENIZERS, f"{tokenizer_type}, {TOKENIZERS}"
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = TOKENIZERS[tokenizer_type]
        # self.tokenizer = tokenizer_none

    @classmethod
    def remove_punctuation(cls, sent: str):
        """Remove punctuation based on Unicode category."""
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


def compute_bleu(refs, hyps):
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    refs = [tokenizer.tokenize(english_normalizer(ref)) for ref in refs]
    hyps = [tokenizer.tokenize(english_normalizer(hyp)) for hyp in hyps]
    # print(f"refs: {refs[0]}")
    # print(f"pred: {hyps[0]}")
    bleu_score = corpus_bleu(hyps, [refs], tokenize="13a")
    return bleu_score.score


def covost2_bleu(results, args):
    refs, hyps = [], []
    for result in results:
        gt = result["gt"]
        response = result["pred"]
        gt = remove_sp(gt)
        response = remove_sp(response)
        refs.append(gt)
        hyps.append(response)
    bleu = compute_bleu(refs, hyps)
    # print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")
    return round(bleu, 5)
