from lmms_eval.api.filter import Filter
from lmms_eval.api.reasoning import strip_reasoning_tags

# Default tag pairs for reasoning models.
_DEFAULT_TAG_PAIRS = [
    ["<think>", "</think>"],
    ["<thinking>", "</thinking>"],
]


class StripThinkingFilter(Filter):
    """Strip reasoning/thinking blocks from model responses.

    Delegates to :func:`lmms_eval.api.reasoning.strip_reasoning_tags` which
    handles both full ``<think>...</think>`` blocks and the common case where
    the chat template injects the opening tag as a prompt prefix (so only
    the closing tag appears in the generated text).

    This filter should run **before** answer-extraction filters (regex, etc.)
    so they see the clean answer text, not the 50 K reasoning chain.

    Args:
        tag_pairs: list of ``[open_tag, close_tag]`` pairs.
            Defaults to ``[["<think>", "</think>"], ["<thinking>", "</thinking>"]]``.
    """

    def __init__(self, tag_pairs: list = None) -> None:
        self.tag_pairs = tag_pairs or _DEFAULT_TAG_PAIRS

    def apply(self, resps, docs):
        def strip_one(text):
            if not isinstance(text, str):
                return text
            return strip_reasoning_tags(text, self.tag_pairs)

        return [[strip_one(r) for r in inst] for inst in resps]


class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = {}, default_value=None) -> None:
        """
        Initializes the MapFilter with a given mapping dictionary and default value.

        Args:
        - mapping_dict (dict): A dictionary containing the key-value mappings.
                               Default is an empty dictionary.
        - default_value (Any): The value to be returned when a key is not found in the mapping_dict.
                               Default is None.

        Example:
        mapper = MapFilter({'A': 1, 'B': 2}, default_value=0)
        """
        assert isinstance(mapping_dict, dict), "Provided mapping_dict is not a dictionary"
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]
