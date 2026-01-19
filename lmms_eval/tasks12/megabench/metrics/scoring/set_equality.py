from metrics.scoring.common.conversions import cast_to_set, str_to_set


class SetEquality:
    """Determines whether two sets are equal."""

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        responses = cast_to_set(responses)
        targets = cast_to_set(targets)
        return 1 if responses == targets else 0


class SetEqualityCaseInsensitive:
    """Determines whether two sets are equal, ignoring string case."""

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        try:
            responses: set[str] = {text.upper() for text in cast_to_set(responses)}
            targets: set[str] = {text.upper() for text in cast_to_set(targets)}
        except AttributeError:
            return 0
        return 1 if responses == targets else 0


class StringSetEqualityLineSplit:
    """Determines whether two sets are equal, for string inputs, separated by line breaks"""

    @classmethod
    def match(cls, responses, targets) -> int:
        if "\\n" in targets:
            targets = targets.replace("\\n", "\n")
        if "\\n" in responses:
            responses = responses.replace("\\n", "\n")
        responses_set = set(responses.split("\n"))
        targets_set = set(targets.split("\n"))
        responses_set = {item.lower() if isinstance(item, str) else item for item in responses_set}
        targets_set = {item.lower() if isinstance(item, str) else item for item in targets_set}
        return 1 if responses_set == targets_set else 0


class StringSetEqualityCommaSplit:
    """Determines whether two sets are equal, for string inputs, separated by commas
    Handles some corner cases that would fail the general SetEquality metric, like the string
    with "None", which fails the eval. Also do case-insensitive eval.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        responses_set = str_to_set(responses)
        targets_set = str_to_set(targets)
        responses_set = {item.lower() if isinstance(item, str) else item for item in responses_set}
        targets_set = {item.lower() if isinstance(item, str) else item for item in targets_set}
        return 1 if responses_set == targets_set else 0
