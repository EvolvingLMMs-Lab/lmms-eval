from enum import Enum
from functools import cached_property

from metrics.aggregation.mean_agg import MeanAggregation
from metrics.aggregation.min_agg import MinAggregation
from metrics.aggregation.unsupported_agg import UnsupportedAggregation


class AggregationType(Enum):
    """The query score aggregation method.

    Enables custom aggregation methods for the field scores.
    """

    MEAN = "mean"
    MIN = "min"
    UNSUPPORTED = "unsupported"

    @cached_property
    def class_impl(self):
        if self == self.MEAN:
            return MeanAggregation
        elif self == self.MIN:
            return MinAggregation
        else:
            return UnsupportedAggregation

    def aggregate(self, scores, weights):
        """Aggregate the field scores."""
        return self.class_impl.aggregate(scores, weights)

    @classmethod
    def from_string(cls, s):
        """Initialize the aggregation type from a string."""
        try:
            if s is None:
                return cls("unsupported")
            return cls(s.lower())
        except KeyError as exc:
            raise ValueError(f"Invalid metric type: {s}") from exc
