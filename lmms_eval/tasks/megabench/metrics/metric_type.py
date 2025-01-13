import logging
from enum import Enum
from functools import cached_property

from metrics.scoring.ascii_art_vlm_judge import AsciiArtVLMJudgeScore
from metrics.scoring.chess_jaccard import ChessMoveJaccard
from metrics.scoring.constrained_generation import ConstrainedGenerationEval
from metrics.scoring.coordinate_sequence_match import CoordsSequenceSimilarity
from metrics.scoring.dict_equality import DictEquality, DictPrecision
from metrics.scoring.dict_exact_match_agg_recall import DictExactStrMatchAggRecall
from metrics.scoring.dict_jaccard_agg_jaccard import DictJaccardAggJaccard
from metrics.scoring.dict_nbbox_iou_tuple_agg_jaccard import DictNbboxIouTupleAggJaccard
from metrics.scoring.dict_set_equality_agg_jaccard import DictSetEqualityAggJaccard
from metrics.scoring.exact_str_match import CodeResultExactStrMatch, ExactStrMatch
from metrics.scoring.exact_str_match_case_insensitive import ExactStrMatchCaseInsensitive
from metrics.scoring.general_numerical_match import BoxedSingleNumericalMatch, GeneralSingleNumericalMatch
from metrics.scoring.geo_proximity import GeoProximityLocationDict
from metrics.scoring.gleu import GLEUChinese
from metrics.scoring.jaccard import Jaccard, JaccardCaseInsensitive
from metrics.scoring.latex_expr_equality import LatexExprEquality, TextLatexExprEquality
from metrics.scoring.longest_common_list_prefix_ratio import LongestCommonListPrefixRatio
from metrics.scoring.mse import AngleSeqFloatRMSE, NormalizedRMSE
from metrics.scoring.multi_ref_phrase import MultipleReferencePhraseEval
from metrics.scoring.nbbox_iou import NbboxIouSequence, NbboxIouSingle, NbboxIouTuple
from metrics.scoring.near_str_match import NearStrMatch
from metrics.scoring.nli_entailment import NliEntailment
from metrics.scoring.normalized_similarity_damerau_levenshtein import NormalizedSimilarityDamerauLevenshtein
from metrics.scoring.number_rel_diff_ratio import NumberRelDiffRatio
from metrics.scoring.positive_int_match import PositiveIntMatch
from metrics.scoring.program_judge import ProgramJudge
from metrics.scoring.sacrebleu_bleu import Bleu
from metrics.scoring.sequence_equality import SequenceAccuracyCaseInsensitive, SequenceEquality, SequenceEqualityCaseInsensitive
from metrics.scoring.set_equality import SetEquality, SetEqualityCaseInsensitive, StringSetEqualityCommaSplit, StringSetEqualityLineSplit
from metrics.scoring.set_precision import SetPrecision

# Import all metrics
from metrics.scoring.simple_str_match import SimpleStrMatch
from metrics.scoring.symbolic_planning import SymbolicPlanningMetricTest
from metrics.scoring.unsupported_scoring import UnsupportedScoring

## The vlm-judge metrics
from metrics.scoring.vlm_as_judge import VLMJudgeScore
from metrics.scoring.xml_nbbox_iou import XmlNbboxIouSingle
from metrics.scoring.xml_norm_point_distance import XmlNormPointDistance
from metrics.scoring.xml_norm_point_in_bbox import XmlNormPointInBbox


class MetricType(Enum):
    """The types of metrics."""

    EXACT_STR_MATCH = "exact_str_match"
    SIMPLE_STR_MATCH = "simple_str_match"
    CODE_RESULT_EXACT_STR_MATCH = "code_result_exact_str_match"
    DICT_EXACT_STR_MATCH_AGG_RECALL = "dict_exact_str_match_agg_recall"
    EXACT_STR_MATCH_CASE_INSENSITIVE = "exact_str_match_case_insensitive"
    NORM_SIM_DAMERAU_LEVENSHTEIN = "normalized_similarity_damerau_levenshtein"
    NEAR_STR_MATCH = "near_str_match"
    NUMBER_RELATIVE_DIFF_RATIO = "number_rel_diff_ratio"
    SET_EQUALITY = "set_equality"
    SET_EQUALITY_CASE_INSENSITIVE = "set_equality_case_insensitive"
    DICT_SET_EQUALITY_AGG_JACCARD = "dict_set_equality_agg_jaccard"
    DICT_PRECISION = "dict_precision"
    JACCARD_INDEX = "jaccard_index"
    JACCARD_INDEX_CASE_INSENSITIVE = "jaccard_index_case_insensitive"
    DICT_JACCARD_AGG_JACCARD = "dict_jaccard_agg_jaccard"
    DICT_EQUALITY = "dict_equality"
    SET_PRECISION = "set_precision"
    POSITIVE_INT_MATCH = "positive_int_match"
    CHESS_MOVE_LIST_JACCARD_INDEX = "chess_move_list_jaccard_index"
    LONGEST_COMMON_LIST_PREFIX_RATIO = "longest_common_list_prefix_ratio"
    NLI_ENTAILMENT = "nli_entailment"
    BLEU = "bleu"
    GLEU_CN = "gleu_cn"
    XML_NORM_BBOX_IOU_SINGLE = "xml_nbbox_iou_single"
    LATEX_EXPR_EQUALITY = "latex_expr_equality"
    TEXT_WITH_LATEX_EXPR_EQUALITY = "text_with_latex_expr_equality"
    NORM_BBOX_IOU_TUPLE = "nbbox_iou_tuple"
    NORM_BBOX_IOU_SINGLE = "nbbox_iou_single"
    NORM_BBOX_IOU_SEQUENCE = "nbbox_iou_sequence"
    DICT_NORM_BBOX_IOU_TUPLE_AGG_JACCARD = "dict_nbbox_iou_tuple_agg_jaccard"
    XML_NORM_POINT_IN_BBOX = "xml_norm_point_in_bbox"
    XML_NORM_POINT_DISTANCE = "xml_norm_point_distance"
    GEO_PROXIMITY_LOCATION_DICT = "geo_proximity_location_dict"
    NORMALIZED_RMSE = "normalized_rmse"
    PROGRAM_JUDGE = "program_judge"
    STR_SET_EQUALITY_LINE_BREAK = "str_set_equality_line_break"
    STR_SET_EQUALITY_COMMA = "str_set_equality_comma"
    SEQUENCE_EQUALITY = "sequence_equality"
    SEQUENCE_EQUALITY_CASE_INSENSITIVE = "sequence_equality_case_insensitive"
    SEQUENCE_ACCURACY_CASE_INSENSITIVE = "sequence_accuracy_case_insensitive"
    ANGLE_SEQ_FLOAT_RMSE = "angle_seq_float_rmse"
    SYMBOLIC_PLANNING_TEST = "symbolic_planning_test"
    MULTI_REF_PHRASE_EVAL = "multi_ref_phrase"
    GENERAL_SINGLE_NUMERICAL_MATCH = "general_single_numerical_match"
    BOXED_SINGLE_NUMERICAL_MATCH = "boxed_single_numerical_match"
    SEQUENCE_COORDS_SIMILARITY = "sequence_coords_similarity"
    CONSTRAINED_GENERATION = "constrained_generation"
    VLM_AS_JUDGE = "gpt_4o_as_judge"
    ASCII_ART_VLM_JUDGE = "ascii_art_gpt4o_judge"
    UNSUPPORTED = "unsupported"

    @cached_property
    def class_impl(self):
        implementations = {
            MetricType.SIMPLE_STR_MATCH: SimpleStrMatch,
            MetricType.EXACT_STR_MATCH: ExactStrMatch,
            MetricType.CODE_RESULT_EXACT_STR_MATCH: CodeResultExactStrMatch,
            MetricType.DICT_EXACT_STR_MATCH_AGG_RECALL: DictExactStrMatchAggRecall,
            MetricType.EXACT_STR_MATCH_CASE_INSENSITIVE: ExactStrMatchCaseInsensitive,
            MetricType.NORM_SIM_DAMERAU_LEVENSHTEIN: NormalizedSimilarityDamerauLevenshtein,
            MetricType.NEAR_STR_MATCH: NearStrMatch,
            MetricType.NUMBER_RELATIVE_DIFF_RATIO: NumberRelDiffRatio,
            MetricType.SET_EQUALITY: SetEquality,
            MetricType.SET_EQUALITY_CASE_INSENSITIVE: SetEqualityCaseInsensitive,
            MetricType.DICT_SET_EQUALITY_AGG_JACCARD: DictSetEqualityAggJaccard,
            MetricType.DICT_EQUALITY: DictEquality,
            MetricType.DICT_PRECISION: DictPrecision,
            MetricType.JACCARD_INDEX: Jaccard,
            MetricType.JACCARD_INDEX_CASE_INSENSITIVE: JaccardCaseInsensitive,
            MetricType.DICT_JACCARD_AGG_JACCARD: DictJaccardAggJaccard,
            MetricType.SET_PRECISION: SetPrecision,
            MetricType.POSITIVE_INT_MATCH: PositiveIntMatch,
            MetricType.CHESS_MOVE_LIST_JACCARD_INDEX: ChessMoveJaccard,
            MetricType.LONGEST_COMMON_LIST_PREFIX_RATIO: LongestCommonListPrefixRatio,
            MetricType.NLI_ENTAILMENT: NliEntailment,
            MetricType.BLEU: Bleu,
            MetricType.GLEU_CN: GLEUChinese,
            MetricType.XML_NORM_BBOX_IOU_SINGLE: XmlNbboxIouSingle,
            MetricType.BOXED_SINGLE_NUMERICAL_MATCH: BoxedSingleNumericalMatch,
            MetricType.GENERAL_SINGLE_NUMERICAL_MATCH: GeneralSingleNumericalMatch,
            MetricType.SEQUENCE_COORDS_SIMILARITY: CoordsSequenceSimilarity,
            MetricType.LATEX_EXPR_EQUALITY: LatexExprEquality,
            MetricType.TEXT_WITH_LATEX_EXPR_EQUALITY: TextLatexExprEquality,
            MetricType.NORM_BBOX_IOU_TUPLE: NbboxIouTuple,
            MetricType.NORM_BBOX_IOU_SINGLE: NbboxIouSingle,
            MetricType.NORM_BBOX_IOU_SEQUENCE: NbboxIouSequence,
            MetricType.DICT_NORM_BBOX_IOU_TUPLE_AGG_JACCARD: DictNbboxIouTupleAggJaccard,
            MetricType.XML_NORM_POINT_IN_BBOX: XmlNormPointInBbox,
            MetricType.XML_NORM_POINT_DISTANCE: XmlNormPointDistance,
            MetricType.GEO_PROXIMITY_LOCATION_DICT: GeoProximityLocationDict,
            MetricType.NORMALIZED_RMSE: NormalizedRMSE,
            MetricType.PROGRAM_JUDGE: ProgramJudge,
            MetricType.STR_SET_EQUALITY_LINE_BREAK: StringSetEqualityLineSplit,
            MetricType.STR_SET_EQUALITY_COMMA: StringSetEqualityCommaSplit,
            MetricType.SEQUENCE_EQUALITY: SequenceEquality,
            MetricType.SEQUENCE_EQUALITY_CASE_INSENSITIVE: SequenceEqualityCaseInsensitive,
            MetricType.SEQUENCE_ACCURACY_CASE_INSENSITIVE: SequenceAccuracyCaseInsensitive,
            MetricType.ANGLE_SEQ_FLOAT_RMSE: AngleSeqFloatRMSE,
            MetricType.SYMBOLIC_PLANNING_TEST: SymbolicPlanningMetricTest,
            MetricType.MULTI_REF_PHRASE_EVAL: MultipleReferencePhraseEval,
            MetricType.CONSTRAINED_GENERATION: ConstrainedGenerationEval,
            MetricType.VLM_AS_JUDGE: VLMJudgeScore,
            MetricType.ASCII_ART_VLM_JUDGE: AsciiArtVLMJudgeScore,
        }

        if self not in implementations:
            logging.error(f"Metric {self} not implemented...")
            return UnsupportedScoring()

        return implementations[self]

    def match(self, response: str, correct_answer: str):
        return self.class_impl.match(response, correct_answer)

    @classmethod
    def from_string(cls, s):
        try:
            if s is None:
                return cls("unsupported")
            return cls(s.lower())
        except KeyError as exc:
            raise ValueError(f"Invalid metric type: {s}") from exc

    @classmethod
    def get_all_values(cls):
        return list(cls)


# List all of the supported metrics:
if __name__ == "__main__":
    print("All MetricType values:")
    for metric_type in MetricType.get_all_values():
        print(f"{metric_type.name}: {metric_type.value}")
