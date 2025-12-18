"""
Processor class for Emu3.5 with IBQ vision tokenizer.
Standalone implementation based on Emu3Processor but specifically for IBQ.
"""

from typing import List, Optional, Sequence

from PIL import Image
import torch
import numpy as np
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging

logger = logging.get_logger(__name__)


def smart_resize(image: Image.Image, area: int = 512 * 512, ds_factor: int = 16):
    width, height = image.size
    aspect_ratio = width / height
    new_height = int((area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    # Round to nearest multiple of divisible_by
    new_height = ((new_height + ds_factor//2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor//2) // ds_factor) * ds_factor
    return image.resize((new_width, new_height), Image.BICUBIC)


def format_image_string(tokenizer, image_tokens):
    image_string = ""
    h, w = image_tokens.shape
    for _h in range(h):
        row_string = ""
        for _w in range(w):
            row_string += "<|visual token {token_id:0>6d}|>".format(token_id=image_tokens[_h, _w])

        if _h < h - 1:
            row_string += tokenizer.eol_token
        image_string += row_string

    return "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}".format(
        image_start=tokenizer.boi_token,
        token_height=h,
        token_width=w,
        image_token=tokenizer.img_token,
        token_str=image_string,
        image_end=tokenizer.eoi_token,
    )


@torch.no_grad()
def build_image(image, img_area, tokenizer, vq_model):
    image = smart_resize(image, img_area)
    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image = torch.tensor((np.array(image) / 127.5 - 1.0)).to(device, dtype).permute(2, 0, 1)
    _, _, token = vq_model.encode(image[None])
    token = token[-1].view(h // 16, w // 16)
    return format_image_string(tokenizer, token)


class Emu3p5Processor:
    r"""
    Processor for Emu3.5 with IBQ (Index Back Quantization) vision tokenizer.

    Args:
        image_processor ([`Emu3_5VisionVQImageProcessor`]):
            The image processor for Emu3.5.
        vision_tokenizer (IBQ):
            The IBQ vision tokenizer for Emu3.5.
        tokenizer ([`Emu3Tokenizer`]):
            The text tokenizer.
    """

    def __init__(
        self,
        vision_tokenizer=None,
        tokenizer=None,
        min_pixels: int = 512 * 512,
        max_pixels: int = 1024 * 1024,
        **kwargs,
    ):
        assert vision_tokenizer is not None, "image tokenizer can not be None"
        assert tokenizer is not None, "trxt tokenizer cant be None"

        self.vision_tokenizer = vision_tokenizer
        self.txt_tokenizer = tokenizer
        # EMU3.5 Is a generative model by design, not meant for understanding tasks, nevertheless try this format.
        self.chat_template = "<|extra_203|>You are a helpful assistant. USER: {question}{images} ASSISTANT: <|extra_100|>"
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels


        # Cache device and dtype from vision tokenizer (required for accelerator-wrapped models)
        self._vt_device = next(self.vision_tokenizer.parameters()).device
        self._vt_dtype = next(self.vision_tokenizer.parameters()).dtype


    @torch.no_grad()
    def __call__(
        self,
        text: Optional[TextInput | PreTokenizedInput] = None,
        image: Optional[Image.Image | List[Image.Image]] = None,
        padding_image: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Emu3Tokenizer's [`~Emu3Tokenizer.__call__`] to encode the text.
        To prepare the image(s), this method forwards the `image` argument to
        Emu3_5VisionVQImageProcessor's [`~Emu3_5VisionVQImageProcessor.__call__`] and IBQ's [`~IBQ.encode`]
        if `image` is not `None`. Please refer to the doctsring of the above two methods for more information.

        Args:
            text (`str` or `List[str]`):
                The sequence or a batch of sequence to be encoded. A sequence is a string.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                The image or a batch of images to be prepared. An image is a PIL image.
            padding_image (`bool`, *optional*):
                whether pad images to same size for fast preprocessing if they have different sizes

        Returns:
            Input Ids - encoded prompts (token-ids)
        """
        if isinstance(text, str):
            text = [text]
        if isinstance(image, Image.Image):
            image = [image]
        if not isinstance(text[0], str):
            raise ValueError("`text` must be string or list of string")

        if image is None:
            raise ValueError("Invalid input image. Please provide exactly one PIL.Image.Image per text.")

        if not isinstance(image, Sequence) and not isinstance(image, Image.Image):
            raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

        if isinstance(image, Sequence) and not isinstance(image[0], Image.Image):
            raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")
        
        # Returns list of tokenized images (1 image per prompt)
        image_strings = self.tokenize_image(image)
        if len(text) != len(image_strings):
            raise ValueError("number of image must match number of text prompt")

        prompt_list = []
        for idx, text_prompt in enumerate(text):
            image_prompt = image_strings[idx]
            prompt = self.chat_template.format(images=image_prompt, question=text_prompt)
            prompt_list.append(prompt)

        # Prompt already contains special tokens
        text_inputs = self.txt_tokenizer(prompt_list, padding='longest', return_tensors="pt", add_special_tokens=False) 
        return text_inputs # contains input ids and attention mask

    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        docs = self.txt_tokenizer.batch_decode(*args, **kwargs)
        return docs

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        doc = self.txt_tokenizer.decode(*args, **kwargs)
        return doc

    @torch.no_grad()
    def encode_and_inject_vision_tokens(
        self,
        texts: List[str],
        images: List[List[Image.Image]],
        image_placeholder: str = "<image>",
        **kwargs,
    ) -> BatchFeature:
        """
        Process texts with image placeholders and inject IBQ vision tokens.

        This method accepts texts that already have a chat template applied,
        containing image placeholder markers. It encodes images using IBQ
        and replaces the placeholders with formatted token strings.

        Args:
            texts: List of strings with chat template already applied,
                   containing image_placeholder markers where images should be inserted
            images: List of image lists, one list per text (supports multiple images per text)
            image_placeholder: Token to replace with vision tokens (e.g., "<|image|>", "<image>")
            **kwargs: Tokenization kwargs (return_tensors, padding, etc.)

        Returns:
            BatchFeature with input_ids and attention_mask

        Raises:
            ValueError: If number of placeholders doesn't match number of images
        """
        processed_texts = []

        #iterate through samples one by one and encode images
        for text, img_list in zip(texts, images):
            # Count expected placeholders
            num_placeholders = text.count(image_placeholder)
            if num_placeholders != len(img_list):
                raise ValueError(
                    f"Mismatch: {num_placeholders} placeholders in text "
                    f"but {len(img_list)} images provided"
                )

            # Encode all images for this text
            if len(img_list) > 0:
                # Process each image via IBQ
                for img in img_list:
                    # Calculate target area (same logic as tokenize_image)
                    w, h = img.size
                    curr_area = w * h
                    target_area = max(min(self.max_pixels, curr_area), self.min_pixels)

                    # Build image via IBQ (returns formatted string with special tokens)
                    image_string = build_image(
                        img, target_area, self.txt_tokenizer, self.vision_tokenizer
                    )

                    # Replace first occurrence only to maintain order
                    text = text.replace(image_placeholder, image_string, 1)

            processed_texts.append(text)

        # Tokenize the final texts (add_special_tokens=False since template already applied)
        text_inputs = self.txt_tokenizer(
            processed_texts, add_special_tokens=False, **kwargs
        )
        return BatchFeature(
            data={**text_inputs}, tensor_type=kwargs.get("return_tensors")
        )

    def tokenize_image(self, image: List[Image.Image]):
        """
        Tokenize images using IBQ vision tokenizer. FOR NOW ONLY SEQUENTIALLY SUPPORTED!

        Smart resize of image normalization applied here!

        Return list of image strings.
        """
        image_strings=[]
        for img in image:
            w, h = img.size
            curr_area = w * h
            target_area = max(min(self.max_pixels, curr_area), self.min_pixels)
            image_strings.append(build_image(img, target_area, self.txt_tokenizer, self.vision_tokenizer))

        return image_strings
