from metrics.scoring.common.conversions import cast_to_set
from metrics.scoring.common.metrics import set_precision


class SetPrecision:
    """Calculates the set precision for iterables."""

    @classmethod
    def match(cls, responses, targets) -> float:
        """Exact match between targets and responses."""
        if responses is None:
            return 0
        responses = cast_to_set(responses)
        targets = cast_to_set(targets)

        return set_precision(responses, targets)
