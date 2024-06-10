import re
import torch
import torch.utils.checkpoint

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from .tokenization_mplug_owl import MplugOwlTokenizer

from decord import VideoReader
import numpy as np
from PIL import Image
from lmms_eval.models.model_utils.load_video import read_video_pyav


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])
    return offsets


def load_video(path, num_frames=4):
    '''vr = VideoReader(path, height=224, width=224)
    total_frames = len(vr)
    frame_indices = get_index(total_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        images_group.append(img)
    return images_group'''
    # Change a bit here from the original code
    # I use pyav instead of decord because it is much more safer
    # The operations here are the same, we load video and return a list of PIL Image
    # Load video frames
    video_frames = read_video_pyav(path, num_frm=num_frames)
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if video_frames.shape[-3] != target_h or video_frames.shape[-2] != target_w:
        video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()
        video_frames = torch.nn.functional.interpolate(video_frames, size=(target_h, target_w))
        video_frames = video_frames.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    video_frames = [Image.fromarray(frame) for frame in video_frames]
    if len(video_frames) > num_frames:
        video_frames = video_frames[: num_frames]
    return video_frames


class MplugOwlProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = "MplugOwlTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):

        super().__init__(**kwargs)
        self.tokens_to_generate = 0
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.add_BOS = True

    def __call__(self, text=None, images=None, videos=None, num_frames=4, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = tokenize_prompts(
                prompts=text,
                tokens_to_generate=self.tokens_to_generate,
                add_BOS=self.add_BOS,
                tokenizer=self.tokenizer,
                ignore_dist=True,
                **kwargs,
            )
            # encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)

        if videos is not None:
            video_features = []
            for video in videos:
                video_frames = load_video(video, num_frames)
                video_feature = self.image_processor(video_frames, return_tensors=return_tensors, **kwargs)["pixel_values"]
                video_features.append(video_feature)
            video_features = torch.stack(video_features, dim=0)
            video_features = video_features.permute(0, 2, 1, 3, 4)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        if text is not None and videos is not None:
            encoding["video_pixel_values"] = video_features
            return encoding
        elif text is not None:
            return encoding
        elif images is not None:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)
        else:
            return BatchEncoding(data=dict(video_pixel_values=video_pixel_values), tensor_type=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)


class MplugOwlImageProcessor(CLIPImageProcessor):
    pass


def detokenize_generations(tokens_gpu_tensor, lengths_gpu_tensor, return_segments, tokenizer):
    """Detokenize the generated tokens."""

    prompts_plus_generations = []
    if return_segments:
        prompts_plus_generations_segments = []

    tokens = tokens_gpu_tensor.cpu().numpy().tolist()
    lengths = lengths_gpu_tensor.cpu().numpy().tolist()
    for sequence_tokens, length in zip(tokens, lengths):
        sequence_tokens = sequence_tokens[:length]
        prompts_plus_generations.append(tokenizer.detokenize(sequence_tokens))
        if return_segments:
            from tokenizers.decoders import Metaspace

            if hasattr(tokenizer, "tokenizer"):
                if isinstance(tokenizer.tokenizer.decoder, Metaspace):
                    words = tokenizer.tokenizer.decode(sequence_tokens)
                else:
                    words = []
                    for token in sequence_tokens:
                        word = tokenizer.tokenizer.decoder[token]
                        word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode("utf-8", errors="replace")
                        words.append(word)
                prompts_plus_generations_segments.append(words)
            else:
                words = tokenizer.detokenize(sequence_tokens)
                # else:
                #     words = []
                #     for token in sequence_tokens:
                #         word = tokenizer.tokenizer.decoder[token]
                #         word = bytearray(
                #             [tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                #                 'utf-8', errors='replace')
                #         words.append(word)
                prompts_plus_generations_segments.append(words)

    if return_segments:
        return tokens, prompts_plus_generations, prompts_plus_generations_segments

    return tokens, prompts_plus_generations


def tokenize_prompts(prompts=None, tokens_to_generate=None, add_BOS=None, rank=0, tokenizer=None, ignore_dist=False, **kwargs):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    attention_mask = None
    if ignore_dist or torch.distributed.get_rank() == rank:
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor, attention_mask = _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS, tokenizer, **kwargs)
        # We need the sizes of these tensors for the boradcast
        [
            prompts_tokens_cuda_long_tensor.size(0),  # Batch size
            prompts_tokens_cuda_long_tensor.size(1),
        ]  # Sequence lenght

    return {
        "input_ids": prompts_tokens_cuda_long_tensor,
        "attention_mask": attention_mask,
        # "prompt_length": prompts_length_cuda_long_tensor,
    }


def _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS, tokenizer, **kwargs):
    """Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts
      plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them
      into a 2D tensor.
    """

    # Tokenize all the prompts.
    # if add_BOS:
    #     prompts_tokens = [[tokenizer.bos] + tokenizer.tokenize(prompt)
    #                       for prompt in prompts]
    # else:
    #     prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]

    prompts_tokens = [_tokenize_prompt(prompt, tokenizer, add_BOS, **kwargs) for prompt in prompts]

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len = max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = max_prompt_len + tokens_to_generate
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        prompt_tokens.extend([tokenizer.eos_token_id] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.LongTensor(prompts_tokens)
    prompts_length_tensor = torch.LongTensor(prompts_length)
    attention_mask = torch.zeros(prompts_tokens_tensor.shape[:2])
    for i, l in enumerate(prompts_length_tensor):
        attention_mask[i, :l] = 1
    return prompts_tokens_tensor, prompts_length_tensor, attention_mask


def _tokenize_prompt(prompt, tokenizer, add_BOS=False, media_info={"<image>": 65, "<|video|>": 65}, **kwargs):
    media_tokens = {k: -int(i + 1) for i, k in enumerate(media_info.keys())}
    media_lengths = media_info.copy()

    if add_BOS:
        prompt_chunk = [tokenizer.bos_token_id]
    else:
        prompt_chunk = []

    # Pure Text
    if all([media_token not in prompt for media_token in media_tokens.keys()]):
        enc_chunk = prompt_chunk + tokenizer(prompt, add_special_tokens=False, **kwargs)["input_ids"]

    # Multi-Modal Text
    else:
        enc_chunk = prompt_chunk
        pattern = "|".join(map(re.escape, list(media_tokens.keys())))
        chunk_strs = re.split(f"({pattern})", prompt)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]
        for idx, chunk_str in enumerate(chunk_strs):
            if chunk_str in media_tokens:
                enc_chunk += [media_tokens[chunk_str]] * media_lengths[chunk_str]
            else:
                tmp_chunk = tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                # if idx < len(chunk_strs) - 1: # Last chunk should not have eos
                #     tmp_chunk += [tokenizer.eod_id]
                enc_chunk += tmp_chunk
    return enc_chunk


if __name__ == "__main__":
    pass
