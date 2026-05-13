import re
import unicodedata
from collections.abc import Callable, Collection
from importlib import import_module

import editdistance as ed

DEFAULT_PUNCS = "!,.?;:"


def remove_sp(
    text: str,
    language: str,
    *,
    puncs: str = DEFAULT_PUNCS,
    no_space_languages: Collection[str] = (),
) -> str:
    normalized = re.sub(r"<\|.*?\|>", " ", text)
    normalized = re.sub(r"\s+", r" ", normalized)
    normalized = re.sub(f" ?([{puncs}])", r"\1", normalized)
    normalized = normalized.lstrip(" ")
    if language in no_space_languages:
        normalized = re.sub(r"\s+", r"", normalized)
    return normalized


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

        tokenizers = {
            "none": NoneTokenizer,
            "13a": Tokenizer13a,
            "intl": TokenizerV14International,
            "zh": TokenizerZh,
            "ja-mecab": TokenizerJaMecab,
            "char": TokenizerChar,
        }

        assert tokenizer_type in tokenizers, f"{tokenizer_type}, {tokenizers}"
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = tokenizers[tokenizer_type]

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


def _normalize(
    text: str,
    language: str,
    *,
    yue_languages: Collection[str],
    english_languages: Collection[str],
    chinese_languages: Collection[str],
    yue_converter: Callable[[str, str], str] | None,
    english_normalizer: Callable[[str], str] | None,
    chinese_normalizer: Callable[[str], str] | None,
    basic_normalizer: Callable[[str], str] | None,
) -> str:
    if language in yue_languages and yue_converter is not None:
        text = yue_converter(text, "zh-cn")
    if language in english_languages and english_normalizer is not None:
        text = english_normalizer(text)
    if language in chinese_languages and chinese_normalizer is not None:
        text = chinese_normalizer(text)
    elif basic_normalizer is not None:
        text = basic_normalizer(text)
    return text


def compute_wer(
    refs,
    hyps,
    language: str,
    *,
    yue_languages: Collection[str] = (),
    english_languages: Collection[str] = (),
    chinese_languages: Collection[str] = (),
    char_languages: Collection[str] = (),
    yue_converter: Callable[[str, str], str] | None = None,
    english_normalizer: Callable[[str], str] | None = None,
    chinese_normalizer: Callable[[str], str] | None = None,
    basic_normalizer: Callable[[str], str] | None = None,
) -> float:
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    for i in range(len(refs)):
        ref = _normalize(
            refs[i],
            language,
            yue_languages=yue_languages,
            english_languages=english_languages,
            chinese_languages=chinese_languages,
            yue_converter=yue_converter,
            english_normalizer=english_normalizer,
            chinese_normalizer=chinese_normalizer,
            basic_normalizer=basic_normalizer,
        )
        pred = _normalize(
            hyps[i],
            language,
            yue_languages=yue_languages,
            english_languages=english_languages,
            chinese_languages=chinese_languages,
            yue_converter=yue_converter,
            english_normalizer=english_normalizer,
            chinese_normalizer=chinese_normalizer,
            basic_normalizer=basic_normalizer,
        )
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        if language in char_languages:
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
        if i == 0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
            print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance / ref_length


def zhconv_convert(text: str, target: str) -> str:
    return import_module("zhconv").convert(text, target)
