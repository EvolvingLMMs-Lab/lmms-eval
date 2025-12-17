# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Emu3.
THIS IS A COPY OF https://github.com/baaivision/Emu3/blob/main/emu3/mllm/processing_emu3.py
JUST WITH GENERATION RELATED CODE REMOVED TO KEEP DEPENDENCIES MINIMAL
"""

from math import ceil
import re
from typing import List, Optional, Sequence

from PIL import Image
import torch
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Emu3Processor(ProcessorMixin):
    r"""
    Constructs an Emu3 processor which wraps an Emu3 image processor and an Emu3 vision vq model and an Emu3 tokenizer into a single processor.

    [`Emu3Processor`] offers all the functionalities of [`Emu3VisionVQModel`] and [`Emu3Tokenizer`]. See the
    [`~Emu3Processor.__call__`], [`~Emu3Processor.decode`], [`~Emu3Processor.vision_encode`], [`~Emu3Processor.vision_decode`]
    for more information.

    Args:
        image_processor ([`Emu3VisionVQImageProcessor`]):
            The image processor is a required input.
        vision_tokenizer ([`Emu3VisionVQModel`]):
            The vision tokenizer is a required input.
        tokenizer ([`Emu3Tokenizer`]):
            The tokenizer is a required input.
        prefix_template(`str`, *optional*):
            The prefix template for image tokens
        visual_template(`Tuple[str, ...]`, *optional*):
            The visual token template for image tokens
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["vision_tokenizer", "prefix_template", "visual_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        vision_tokenizer=None,
        tokenizer=None,
        chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:",
        prefix_template="{H}*{W}",
        visual_template=("<|visual token {token_id:0>6d}|>", r"<\|visual token (\d+)\|>"),
        **kwargs,
    ):
        assert vision_tokenizer is not None, "image tokenizer can not be None"

        self.vision_tokenizer = vision_tokenizer
        self.prefix_template = prefix_template
        self.visual_template = visual_template
        self.vis_tok_spatial_factor = 2 ** (len(self.vision_tokenizer.config.ch_mult) - 1)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @torch.no_grad()
    def __call__(
        self,
        text: Optional[TextInput | PreTokenizedInput] = None,
        image: Optional[Image.Image | List[Image.Image]] = None,
        *,
        mode: str = "G",
        ratio: str | List[str] = "1:1",
        image_area: int = 518400,
        padding_image: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Emu3Tokenizer's [`~Emu3Tokenizer.__call__`] to encode the text.
        To prepare the image(s), this method forwards the `image` argument to
        Emu3VisionVQImageProcessor's [`~Emu3VisionVQImageProcessor.__call__`] and Emu3VisionVQModel's [`~EmuVideoVQModel.encode`]
        if `image` is not `None`. Please refer to the doctsring of the above two methods for more information.

        Args:
            text (`str` or `List[str]`):
                The sequence or a batch of sequence to be encoded. A sequence is a string.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                The image or a batch of images to be prepared. An image is a PIL image.
            mode (`str`, *optional*, in `G` or `U`):
                task mode, `G` for generation and `U` for understanding
            ratio (`str`, *optional*):
                the image width-height ratio for generation
            image_area (`int`, *optional*):
                image area used to calcualte the generated image height and width
            padding_image (`bool`, *optional*):
                whether pad images to same size for fast preprocessing if they have different sizes
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **image_size** -- List of image size of input images or generated images.
        """
        assert mode in ('G', 'U'), "mode must be 'G' or 'U'."
        if isinstance(text, str):
            text = [text]

        if isinstance(image, Image.Image):
            image = [image]

        if not isinstance(text[0], str):
            raise ValueError("`text` must be string or list of string")

        image_tokens = None
        if mode == 'G':
            raise NotImplementedError("Mode G is not implemented in this adapted Processor version")
        else:
            if image is None:
                raise ValueError("Invalid input image. Please provide exactly one PIL.Image.Image per text.")

            if not isinstance(image, Sequence) and not isinstance(image, Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            if isinstance(image, Sequence) and not isinstance(image[0], Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            image_tokens = self.tokenize_image(image, padding_image=padding_image)
            if len(text) != len(image_tokens):
                raise ValueError("number of image must match number of text prompt")

        prompt_list, size_list = [], []
        for idx, text_prompt in enumerate(text):
            prompt = self.tokenizer.bos_token
            if mode == 'U':
                h, w = image_tokens[idx].shape
                imgstr = self.to_imgstr(image_tokens[idx])
                image_prompt = (
                    self.tokenizer.boi_token +
                    self.prefix_template.format(H=h, W=w) +
                    self.tokenizer.img_token +
                    imgstr +
                    self.tokenizer.eol_token +
                    self.tokenizer.eof_token +
                    self.tokenizer.eoi_token
                )
                prompt += self.chat_template.format(image_prompt=image_prompt, text_prompt=text_prompt)
            else:
                raise NotImplementedError("Invalid Mode: generation mode not supported by this adapted processor")

            prompt_list.append(prompt)
            size_list.append([h, w])

        text_inputs = self.tokenizer(prompt_list, **kwargs)
        return BatchFeature(data={**text_inputs, "image_size": size_list}, tensor_type=kwargs.get("return_tensors"))

    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        docs = self.tokenizer.batch_decode(*args, **kwargs)
        return docs

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        doc = self.tokenizer.decode(*args, **kwargs)
        return doc

    @torch.no_grad()
    def vision_encode(self, *args, **kwargs):
        return self.vision_tokenizer.encode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def to_imgstr(self, image_tokens):
        image_tokens = image_tokens.cpu().numpy().tolist()
        image_token_str = [
            [
                self.visual_template[0].format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

    def tokenize_image(self, image: List[Image.Image], *, padding_image: bool = False):
        is_all_same_size, prev_size = True, None
        for im in image:
            if prev_size is not None:
                is_all_same_size &= (prev_size == im.size)
            prev_size = im.size

        if is_all_same_size:
            image_inputs = self.image_processor(image, return_tensors="pt")["pixel_values"]
            image_inputs = image_inputs.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
        elif padding_image:
            image_inputs = [self.image_processor(im, return_tensors="pt")["pixel_values"] for im in image]
            image_shapes = [im.shape[2:] for im in image_inputs]
            max_shape = (
                max([im_shape[0] for im_shape in image_shapes]),
                max([im_shape[1] for im_shape in image_shapes]),
            )
            image_inputs = [
                F.pad(im_inp, (0, max_shape[1] - im_shape[1], 0, max_shape[0] - im_shape[0]))
                for im_inp, im_shape in zip(image_inputs, image_shapes)
            ]
            image_inputs = torch.cat(image_inputs, dim=0).to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
            image_tokens = [
                im_tok[:ceil(im_shape[0] / self.vis_tok_spatial_factor), :ceil(im_shape[1] / self.vis_tok_spatial_factor)]
                for im_tok, im_shape in zip(image_tokens, image_shapes)
            ]
        else:
            image_tokens = []
            for im in image:
                image_input = self.image_processor(im, return_tensors="pt")["pixel_values"]
                image_input = image_input.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
                image_tokens.append(self.vision_tokenizer.encode(image_input).squeeze(0))

        return image_tokens
