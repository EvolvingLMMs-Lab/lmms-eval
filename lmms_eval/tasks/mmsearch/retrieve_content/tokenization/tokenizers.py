import logging
import re
import sys
import unicodedata
from typing import Any, Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize as sent_tok

from lmms_eval.tasks.mmsearch.retrieve_content.tokenization.utils import PickleWriteable

QUOTES = re.compile("(\"|``|'')")


class LexemeWithPositions(dict):
    def __init__(self, lexeme: str, begin_pos: int, end_pos: int):
        if begin_pos > end_pos:
            raise ValueError("begin position cannot be greater than end position for a lexeme")
        super().__init__(_lexeme=lexeme, _begin_pos=begin_pos, _end_pos=end_pos)

    def fetch_lexeme(self) -> str:
        return self["_lexeme"]

    def update_lexeme(self, lex: str) -> None:
        self["_lexeme"] = lex

    def fetch_begin_pos(self) -> int:
        return self["_begin_pos"]

    def fetch_end_pos(self) -> int:
        return self["_end_pos"]


class LexicalAnalyzer(PickleWriteable):
    PUNCTUATION_TABLE = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))
    PUNCTUATION_CHARS = set(chr(i) for i in PUNCTUATION_TABLE.keys())

    DEFAULT_SETTINGS = {
        "min_lexeme_len": 1,
        "max_lexeme_len": 25,
        "space_chars": [],
        "do_stop_words": False,
        "excerpt_len": 200,
        "max_sent_len": 35,
        "respect_sent_boundaries": False,
        "do_sliding_window_excerpts": False,
        "do_char_positions": False,
        "strip_inlexeme_punctuation": False,
        "remove_numerics": False,
        "do_stem": False,
        "do_lemma": False,
        "do_punct_removal": False,
        "do_lower": False,
        "lexeme_splitter": nltk.tokenize.word_tokenize,
        "reject_threshold": -1,
    }

    def __init__(self, **kwargs):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.settings.update(kwargs)

        if self.settings["do_stop_words"]:
            self.stop_words = set(self.settings.get("stop_words", stopwords.words("english")))
        elif "stop_words" in kwargs:
            raise ValueError("please set do_stop_words to True when stop_words are provided.")

        if self.settings["do_stem"] and self.settings["do_lemma"]:
            raise ValueError("stemming and lemmatization cannot be turned on simultaneously")

        self.stemmer = PorterStemmer() if self.settings["do_stem"] else None
        self.lemmatizer = WordNetLemmatizer() if self.settings["do_lemma"] else None

    def analyze_excerpts(self, text: str) -> List[List[Any]]:
        if self.settings["respect_sent_boundaries"]:
            return self._analyze_excerpts_with_sentence_boundaries(text)
        return self._analyze_excerpts_without_sentence_boundaries(text)

    def _analyze_excerpts_with_sentence_boundaries(self, text: str) -> List[List[List[LexemeWithPositions]]]:
        analyzed_sentences = self.analyze_sentences(text)
        analyzed_sentences = self._split_long_sentences(analyzed_sentences)

        if not self.settings["do_sliding_window_excerpts"]:
            return self._generate_nonoverlapping_excerpts(analyzed_sentences)

        nonoverlapping_excerpts = self._generate_nonoverlapping_excerpts(analyzed_sentences)
        if len(nonoverlapping_excerpts) <= 1:
            return nonoverlapping_excerpts

        return self._generate_overlapping_excerpts(nonoverlapping_excerpts)

    def _analyze_excerpts_without_sentence_boundaries(self, text: str) -> List[List[LexemeWithPositions]]:
        analyzed_lexemes = self.analyze(text)
        if self.settings["do_sliding_window_excerpts"]:
            return self._generate_sliding_window_excerpts(analyzed_lexemes)
        return _split_analyzed_text_to_nonoverlapping_excerpts(analyzed_lexemes, self.settings["excerpt_len"])

    def _generate_nonoverlapping_excerpts(self, analyzed_sentences: List[List[LexemeWithPositions]]) -> List[List[List[LexemeWithPositions]]]:
        excerpts = []
        current_excerpt = []
        current_excerpt_len = 0
        sent_lens = [len(sent) for sent in analyzed_sentences]

        for i, sent in enumerate(analyzed_sentences):
            current_excerpt.append(sent)
            if i < len(analyzed_sentences) - 1 and sum(sent_lens[i + 1 :]) <= self.settings["excerpt_len"] / 2:
                current_excerpt.extend(analyzed_sentences[i + 1 :])
                excerpts.append(current_excerpt)
                break
            current_excerpt_len += len(sent)
            if current_excerpt_len >= self.settings["excerpt_len"] or i == len(analyzed_sentences) - 1:
                excerpts.append(current_excerpt)
                current_excerpt = []
                current_excerpt_len = 0
        return excerpts

    def _generate_overlapping_excerpts(self, nonoverlapping_excerpts: List[List[List[LexemeWithPositions]]]) -> List[List[List[LexemeWithPositions]]]:
        overlapping_excerpts = []
        for i in range(len(nonoverlapping_excerpts) - 1):
            left_excerpt_start_index = len(nonoverlapping_excerpts[i]) // 2
            right_excerpt_end_index = max(1, len(nonoverlapping_excerpts[i + 1]) // 2)
            overlapping_excerpts.append(nonoverlapping_excerpts[i][left_excerpt_start_index:] + nonoverlapping_excerpts[i + 1][:right_excerpt_end_index])

        excerpts = []
        for i in range(len(nonoverlapping_excerpts) - 1):
            excerpts.append(nonoverlapping_excerpts[i])
            excerpts.append(overlapping_excerpts[i])
        excerpts.append(nonoverlapping_excerpts[-1])
        return excerpts

    def _generate_sliding_window_excerpts(self, analyzed_lexemes: List[LexemeWithPositions]) -> List[List[LexemeWithPositions]]:
        excerpts = []
        for i in range(0, len(analyzed_lexemes), self.settings["excerpt_len"] // 2):
            excerpts.append(analyzed_lexemes[i : i + self.settings["excerpt_len"]])
            if i + self.settings["excerpt_len"] >= len(analyzed_lexemes):
                break
        return excerpts

    def analyze_sentences(self, text: str) -> List[List[LexemeWithPositions]]:
        analyzed_sentences = []
        for sent_with_positions in self._analyze_text(text, sent_tok):
            sent_text = sent_with_positions.fetch_lexeme()
            sent_start = sent_with_positions.fetch_begin_pos()
            analyzed_sentences.append(self.analyze(sent_text, sent_start))
        return analyzed_sentences

    def analyze(self, text: str, offset: int = 0) -> List[LexemeWithPositions]:
        analyzed_lexemes = self._analyze_text(text, self.settings["lexeme_splitter"], offset)
        transformed_lexemes = self._filter_and_transform(analyzed_lexemes)
        if not self.settings["do_char_positions"]:
            return [lexeme_with_positions.fetch_lexeme() for lexeme_with_positions in transformed_lexemes]
        return transformed_lexemes

    def _split_long_sentences(self, analyzed_sentences: List[List[LexemeWithPositions]]) -> List[List[LexemeWithPositions]]:
        split_sentences = []
        for analyzed_sent in analyzed_sentences:
            if len(analyzed_sent) > self.settings["max_sent_len"]:
                sent_excerpts = _split_analyzed_text_to_nonoverlapping_excerpts(analyzed_sent, self.settings["max_sent_len"])
                split_sentences.extend(sent_excerpts)
            else:
                split_sentences.append(analyzed_sent)
        return split_sentences

    def _analyze_text(self, text: str, analyzer: callable, offset: int = 0) -> List[LexemeWithPositions]:
        if not isinstance(text, str):
            raise ValueError(f"text type is invalid: {type(text)}")

        for space_char in self.settings["space_chars"]:
            text = text.replace(space_char, " ")

        text = self._omit_long_lexemes(text, self.settings["reject_threshold"])
        segments = analyzer(text)

        if analyzer is nltk.word_tokenize:
            segments = self._divide_long_lexemes(segments)

        analyzed_segments = []
        start = 0
        for segment in segments:
            segment_start = text.find(segment, start)
            if analyzer is nltk.word_tokenize and segment in ["``", "''"]:
                quotes_match = QUOTES.search(text, start)
                if quotes_match:
                    segment = quotes_match.group(0)
                    segment_start = quotes_match.start()
                else:
                    segment_start = -1

            if segment_start < 0:
                raise ValueError(f"cannot find the segment {segment} in the text {text}")

            end = segment_start + len(segment)
            analyzed_segments.append(LexemeWithPositions(segment, offset + segment_start, offset + end))
            start = end

        return analyzed_segments

    def _divide_long_lexemes(self, segments: List[str]) -> List[str]:
        divided_segments = []
        for seg in segments:
            if len(seg) <= self.settings["max_lexeme_len"]:
                divided_segments.append(seg)
            else:
                divided_segments.extend([seg[i : i + self.settings["max_lexeme_len"]] for i in range(0, len(seg), self.settings["max_lexeme_len"])])
        return divided_segments

    def _omit_long_lexemes(self, text: str, reject_threshold: int) -> str:
        if reject_threshold < 0:
            return text

        verified_sentence = []
        omitted_count = 0
        for lexeme in text.split():
            if len(lexeme) < reject_threshold:
                verified_sentence.append(lexeme)
            else:
                omitted_count += 1
                logging.error(f"omitting long lexeme with length: {len(lexeme)}")

        logging.info(f"total omitted: {omitted_count}, total retained: {len(verified_sentence)}")
        return " ".join(verified_sentence)

    def _filter_and_transform(self, lexemes_with_positions: List[LexemeWithPositions]) -> List[LexemeWithPositions]:
        transformed_lexemes = []

        for lexeme_w_positions in lexemes_with_positions:
            lexeme = lexeme_w_positions.fetch_lexeme()

            if self._should_omit_lexeme(lexeme):
                continue

            lexeme = self._transform_lexeme(lexeme)

            if lexeme is None:
                logging.error("analysis produced None as a lexeme")
                continue

            lexeme_w_positions.update_lexeme(lexeme)
            transformed_lexemes.append(lexeme_w_positions)

        return transformed_lexemes

    def _should_omit_lexeme(self, lexeme: str) -> bool:
        return (
            (self.settings["do_punct_removal"] and lexeme in self.PUNCTUATION_CHARS)
            or (self.settings["do_stop_words"] and lexeme.lower() in self.stop_words)
            or (len(lexeme) < self.settings["min_lexeme_len"])
            or (self.settings["remove_numerics"] and lexeme.isnumeric())
        )

    def _transform_lexeme(self, lexeme: str) -> str:
        if self.settings["do_lower"]:
            lexeme = lexeme.lower()
        if self.settings["strip_inlexeme_punctuation"]:
            lexeme = lexeme.translate(self.PUNCTUATION_TABLE)
        if self.settings["do_stem"]:
            lexeme = self.stemmer.stem(lexeme)
        elif self.settings["do_lemma"]:
            lexeme = self.lemmatizer.lemmatize(lexeme)
        return lexeme


def _split_analyzed_text_to_nonoverlapping_excerpts(analyzed_lexemes: List[LexemeWithPositions], excerpt_len: int) -> List[List[LexemeWithPositions]]:
    excerpts = []
    for i in range(0, len(analyzed_lexemes), excerpt_len):
        if len(analyzed_lexemes) - (i + excerpt_len) <= excerpt_len / 2:
            excerpts.append(analyzed_lexemes[i:])
            break
        excerpts.append(analyzed_lexemes[i : i + excerpt_len])
    return excerpts
