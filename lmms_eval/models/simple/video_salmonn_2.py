import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import

# The video_SALMONN2_plus model class lives in the source repo's custom module.
# It must be installed from https://github.com/bytedance/video-SALMONN-2
# (the video_SALMONN2_plus/qwenvl package).
video_SALMONN2_plus_cls, _has_salmonn = optional_import(
    "qwenvl.model.modeling_qwen2_5_vl", "video_SALMONN2_plus"
)
PeftModel, _has_peft = optional_import("peft", "PeftModel")
WhisperFeatureExtractor, _has_whisper_fe = optional_import(
    "transformers", "WhisperFeatureExtractor"
)
AutoTokenizer, _ = optional_import("transformers", "AutoTokenizer")
Qwen2_5_VLProcessor, _ = optional_import("transformers", "Qwen2_5_VLProcessor")

process_vision_info, _has_qwen_vl = optional_import(
    "qwen_vl_utils", "process_vision_info"
)
if not _has_qwen_vl:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; "
        "Please install via `pip install qwen-vl-utils`"
    )

librosa, _has_librosa = optional_import("librosa")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 30
# Each 30-second audio chunk produces 60 audio pad tokens after QFormer.
AUDIO_TOKENS_PER_CHUNK = 60
AUDIO_PAD_TOKEN = "<|audio_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"


def _extract_audio_from_video(video_path: str, sr: int = AUDIO_SAMPLE_RATE):
    """Extract audio waveform from a video file using librosa."""
    try:
        audio_array, _ = librosa.load(video_path, sr=sr, mono=True)
        return audio_array
    except Exception as e:
        eval_logger.warning(
            f"Failed to extract audio from {video_path}: {e}. "
            "Proceeding without audio."
        )
        return None


def _process_audio_to_features(
    audio_array: np.ndarray,
    audio_processor: "WhisperFeatureExtractor",
    sr: int = AUDIO_SAMPLE_RATE,
):
    """Convert raw audio waveform to mel-spectrogram features.

    Returns:
        audio_feature: Tensor of shape [num_chunks, feature_dim, time_steps].
        audio_length: Number of audio pad tokens to insert
            (num_chunks * AUDIO_TOKENS_PER_CHUNK).
    """
    chunk_samples = AUDIO_CHUNK_SECONDS * sr
    processor = copy.deepcopy(audio_processor)

    # Split audio into 30-second chunks.
    chunks = [
        audio_array[k : k + chunk_samples]
        for k in range(0, len(audio_array), chunk_samples)
    ]

    spectrogram_list = []
    for chunk in chunks:
        features = processor(
            chunk,
            sampling_rate=sr,
            return_tensors="pt",
        )["input_features"].squeeze(0)
        spectrogram_list.append(features)

    audio_feature = torch.stack(spectrogram_list, dim=0)
    num_chunks = math.ceil(len(audio_array) / chunk_samples)
    audio_length = num_chunks * AUDIO_TOKENS_PER_CHUNK
    return audio_feature, audio_length


@register_model("video_salmonn_2")
class VideoSALMONN2(lmms):
    """video-SALMONN-2+ (7B/72B) -- audio-visual large language model.

    Built on Qwen2.5-VL with additional WhisperEncoder + QFormer + audio
    projection for full audio-visual understanding.

    Source: https://github.com/bytedance/video-SALMONN-2
    HF: https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B

    Usage
    -----
    The ``pretrained`` argument must point to an audio-aligned base model that
    uses the ``video_SALMONN2_plus`` architecture (contains WhisperEncoder,
    QFormer, audio_proj). Generate this base model by running
    ``scripts/gen_audio_model.py`` from the source repo, or use a
    pre-generated checkpoint.

    Optionally, ``lora_ckpt`` can point to a LoRA adapter (e.g., the HF repo
    ``tsinghua-ee/video-SALMONN-2_plus_7B``).  When provided the adapter is
    loaded and merged into the base model before inference.
    """

    def __init__(
        self,
        pretrained: str = "tsinghua-ee/video-SALMONN-2_plus_7B",
        lora_ckpt: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        max_num_frames: int = 768,
        num_video_frames: int = 50,
        max_pixels: int = 61250,
        min_pixels: int = 784,
        audio_visual: bool = True,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if not _has_salmonn:
            raise ImportError(
                "Could not import video_SALMONN2_plus model class. "
                "Please install from https://github.com/bytedance/video-SALMONN-2 "
                "(the video_SALMONN2_plus/qwenvl package must be on sys.path)."
            )

        # ---- device / accelerator setup ----
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(
                f"cuda:{accelerator.local_process_index}"
            )
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(
                f"cuda:{accelerator.local_process_index}"
            )
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # ---- audio-visual flag (set early, used during LoRA loading) ----
        self.audio_visual = audio_visual

        # ---- load the video_SALMONN2_plus model ----
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = video_SALMONN2_plus_cls.from_pretrained(
            pretrained, **model_kwargs
        )

        # ---- optionally apply and merge LoRA adapter ----
        if lora_ckpt is not None and _has_peft:
            eval_logger.info(f"Loading LoRA adapter from {lora_ckpt}")
            # The LoRA config excludes audio.layers from the adapter
            # (modules_to_save handles qformer/q_tokens/audio_proj instead).
            # We must temporarily detach audio.layers so PeftModel does not
            # attempt to wrap them, following the reference implementation.
            if self.audio_visual and hasattr(self._model, "audio"):
                audio_layers = self._model.audio.layers
                del self._model.audio.layers
            self._model = PeftModel.from_pretrained(self._model, lora_ckpt)
            if self.audio_visual and hasattr(self._model.model, "audio"):
                self._model.model.audio.layers = audio_layers
            self._model = self._model.merge_and_unload()

        self._model.eval()

        # ---- processor and tokenizer ----
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        self._tokenizer = self.processor.tokenizer

        # ---- audio processor ----
        if self.audio_visual and _has_whisper_fe:
            self.audio_processor = WhisperFeatureExtractor(
                feature_size=128,
                sampling_rate=AUDIO_SAMPLE_RATE,
                hop_length=160,
                chunk_length=AUDIO_CHUNK_SECONDS,
            )
        else:
            self.audio_processor = None
            if self.audio_visual:
                eval_logger.warning(
                    "WhisperFeatureExtractor not available; "
                    "audio processing will be disabled."
                )

        # ---- configuration ----
        self.max_num_frames = max_num_frames
        self.num_video_frames = num_video_frames
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt

        # ---- distributed setup (keep unchanged) ----
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], (
                "Unsupported distributed type provided. "
                "Only DDP and FSDP are supported."
            )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices "
                    "with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ------------------------------------------------------------------
    # Not implemented
    # ------------------------------------------------------------------

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Loglikelihood is not implemented for VideoSALMONN2"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_audio_tokens(self, audio_length: int) -> str:
        """Return the token string to embed audio in the chat template."""
        return (
            VISION_START_TOKEN
            + AUDIO_PAD_TOKEN * audio_length
            + VISION_END_TOKEN
        )

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in doc_id
            ]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get(
                "until", [self.tokenizer.decode(self.eot_token_id)]
            )
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type "
                    f"Union[str, list], but got {type(until)}"
                )
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            # ---- per-sample: build messages, extract audio ----
            batched_messages = []
            batch_audio_features: List[Optional[torch.Tensor]] = []
            batch_audio_lengths: List[List[int]] = []

            for i, context in enumerate(contexts):
                message = [
                    {"role": "system", "content": self.system_prompt}
                ]
                processed_visuals = []
                audio_features_list: List[torch.Tensor] = []
                audio_lengths_list: List[int] = []
                audio_token_str = ""

                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith(
                            VIDEO_EXTENSIONS
                        ):
                            # Add video to the message for Qwen2.5-VL
                            # vision processing.
                            processed_visuals.append(
                                {
                                    "type": "video",
                                    "video": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )

                            # Extract audio from the video file.
                            if (
                                self.audio_visual
                                and self.audio_processor is not None
                            ):
                                audio_array = _extract_audio_from_video(
                                    visual, sr=AUDIO_SAMPLE_RATE
                                )
                                if audio_array is not None and len(
                                    audio_array
                                ) > 0:
                                    feat, length = (
                                        _process_audio_to_features(
                                            audio_array,
                                            self.audio_processor,
                                            sr=AUDIO_SAMPLE_RATE,
                                        )
                                    )
                                    audio_features_list.append(feat)
                                    audio_lengths_list.append(length)
                                    audio_token_str += (
                                        self._build_audio_tokens(length)
                                    )

                        elif isinstance(visual, Image.Image):
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )

                        elif isinstance(visual, dict) and "array" in visual:
                            # Standalone audio dict from the benchmark.
                            if (
                                self.audio_visual
                                and self.audio_processor is not None
                            ):
                                audio_array = visual["array"]
                                sr = visual.get(
                                    "sampling_rate", AUDIO_SAMPLE_RATE
                                )
                                # Resample if needed.
                                if sr != AUDIO_SAMPLE_RATE and _has_librosa:
                                    audio_array = librosa.resample(
                                        audio_array,
                                        orig_sr=sr,
                                        target_sr=AUDIO_SAMPLE_RATE,
                                    )
                                if isinstance(audio_array, np.ndarray):
                                    feat, length = (
                                        _process_audio_to_features(
                                            audio_array,
                                            self.audio_processor,
                                            sr=AUDIO_SAMPLE_RATE,
                                        )
                                    )
                                    audio_features_list.append(feat)
                                    audio_lengths_list.append(length)
                                    audio_token_str += (
                                        self._build_audio_tokens(length)
                                    )

                        elif isinstance(visual, str) and visual.endswith(
                            AUDIO_EXTENSIONS
                        ):
                            # Standalone audio file path.
                            if (
                                self.audio_visual
                                and self.audio_processor is not None
                                and _has_librosa
                            ):
                                audio_array = _extract_audio_from_video(
                                    visual, sr=AUDIO_SAMPLE_RATE
                                )
                                if audio_array is not None and len(
                                    audio_array
                                ) > 0:
                                    feat, length = (
                                        _process_audio_to_features(
                                            audio_array,
                                            self.audio_processor,
                                            sr=AUDIO_SAMPLE_RATE,
                                        )
                                    )
                                    audio_features_list.append(feat)
                                    audio_lengths_list.append(length)
                                    audio_token_str += (
                                        self._build_audio_tokens(length)
                                    )

                # Build content: visuals + audio tokens + text.
                user_content = processed_visuals[:]
                if audio_token_str:
                    # Prepend audio token placeholder before the text
                    # so the model sees audio context before the question.
                    user_content.append(
                        {"type": "text", "text": audio_token_str}
                    )
                user_content.append({"type": "text", "text": context})

                message.append({"role": "user", "content": user_content})
                batched_messages.append(message)

                # Collect audio features for this sample.
                if audio_features_list:
                    batch_audio_features.append(
                        torch.cat(audio_features_list, dim=0)
                    )
                    batch_audio_lengths.append(audio_lengths_list)
                else:
                    batch_audio_features.append(None)
                    batch_audio_lengths.append([])

            # ---- tokenize text with chat template ----
            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # ---- process vision info ----
            image_inputs, video_inputs = process_vision_info(
                batched_messages
            )

            # Subsample video frames to num_video_frames.
            if video_inputs is not None:
                for vidx in range(len(video_inputs)):
                    total_frames = video_inputs[vidx].shape[0]
                    target = min(self.num_video_frames, total_frames)
                    target = min(self.max_num_frames, target)
                    if target < total_frames:
                        indices = np.linspace(
                            0,
                            total_frames - 1,
                            target,
                            dtype=int,
                            endpoint=True,
                        )
                        indices = np.unique(indices)
                        video_inputs[vidx] = video_inputs[vidx][indices]

            # ---- build model inputs via the Qwen processor ----
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            # Move inputs to device.
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # ---- add audio features to inputs ----
            if self.audio_visual and any(
                f is not None for f in batch_audio_features
            ):
                # Concatenate all audio features across the batch.
                all_audio = [
                    f for f in batch_audio_features if f is not None
                ]
                if all_audio:
                    audio_feature = torch.cat(all_audio, dim=0).to(
                        dtype=torch.bfloat16, device=inputs["input_ids"].device
                    )
                    audio_lengths = []
                    for lengths in batch_audio_lengths:
                        audio_lengths.extend(lengths)
                    inputs["audio_feature"] = audio_feature
                    inputs["audio_lengths"] = audio_lengths

            # ---- generation kwargs ----
            default_gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            # ---- generate ----
            try:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), answer
                )
                continue

            # ---- decode ----
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), ans
                )
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
