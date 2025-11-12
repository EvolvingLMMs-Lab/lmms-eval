from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from lmms_eval import utils


class BaseVideoSampler(ABC):
    """Abstract base class for video samplers used by multimodal models.

    Samplers take a raw ``visual`` item (e.g. a video path or frame list) and
    return a dictionary payload understood by the downstream processor.
    """

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        """
        Creates an instance of the LMM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the video sampler class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    @abstractmethod
    def sample(
        self,
        ele: Any,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Return a processed representation for ``visual``.

        Implementations may return ``None`` to signal that the input should be
        skipped.
        """

    def __call__(
        self,
        ele: Any,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        return self.sample(
            ele,
            **kwargs
        )

