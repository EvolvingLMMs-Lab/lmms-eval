"""
Image processor for Emu3.5 Vision Tokenizer (IBQ).
Wraps the Emu3 image processor with Emu3.5-specific spatial downsample ratio.
https://github.com/baaivision/Emu3.5
"""

from typing import List, Optional, Union

from transformers.image_utils import PILImageResampling

from lmms_eval.models.model_utils.emu3.emu3_image_processor import (
    Emu3VisionVQImageProcessor,
    smart_resize,
)


class Emu3_5VisionVQImageProcessor(Emu3VisionVQImageProcessor):
    """
    Image processor for Emu3.5 IBQ Vision Tokenizer.

    Inherits from Emu3VisionVQImageProcessor but uses a different spatial
    downsample ratio. Emu3.5's IBQ tokenizer uses spatial_factor=16 (due to
    ch_mult=[1,1,2,2,4] with length 5: 2^(5-1)=16), while Emu3 uses 8.
    """

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 512 * 512,
        max_pixels: int = 1024 * 1024,
        do_check_aspect_ratio: bool = False,
        spatial_factor: int = 16,  # Emu3.5 uses 16 (Emu3 uses 8)
        **kwargs,
    ) -> None:
        """
        Initialize Emu3_5VisionVQImageProcessor.

        Args:
            spatial_factor: Spatial downsample factor for Emu3.5 IBQ tokenizer.
                           Default is 16 (ch_mult=[1,1,2,2,4] -> 2^(5-1)=16).
                           This differs from Emu3 which uses 8.
        """
        super().__init__(
            do_resize=do_resize,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            do_check_aspect_ratio=do_check_aspect_ratio,
            spatial_factor=spatial_factor,
            **kwargs,
        )


__all__ = ["Emu3_5VisionVQImageProcessor", "smart_resize"]