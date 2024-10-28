import copy
import json
import logging
import os
import os.path as osp
from typing import List, Optional, Tuple, Union

import av
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from lmms_eval.utils import stop_sequences_criteria

try:
    from lmms_eval.models.aurora_xtuner.model.aurora import (
        AuroraEncoder,
        AuroraModel,
        AuroraSigEncoder,
    )
    from lmms_eval.models.aurora_xtuner.utils import PROMPT_TEMPLATE
except ImportError:
    eval_logger.error("AuroraCap is not installed. Please install AuroraCap to use this model by `git clone https://github.com/rese1f/aurora.git` and link `src/xtuner/xtuner` to `lmms_eval/models/aurora_xtuner`")
import warnings

warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")

try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
except ImportError:
    eval_logger.error("LLaVA is not installed. Please install LLaVA to use this model.")


@register_model("auroracap")
class AuroraCap(lmms):
    """
    auroracap Model
    """

    def __init__(
        self,
        pretrained_llm: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        pretrained_vit: str = "google/siglip-so400m-patch14-384",
        pretrained: str = "model/PATH",
        resolution: int = 378,
        token_merge_ratio: float = 0.4,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        conv_template="vicuna_v1",  # vicuna_v1",
        video_decode_backend: str = "pyav",
        max_frames_num: int = 16,
        slowfast: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        pretrained_pth = snapshot_download(repo_id=pretrained) if not osp.isdir(pretrained) else pretrained
        pretrained_llm = pretrained_pth
        pretrained_vit = osp.join(pretrained_pth, "visual_encoder")

        self._model = AuroraModel(
            slowfast=slowfast,
            llm=AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_llm,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ),
            visual_encoder=AuroraEncoder.from_pretrained(
                pretrained_model_name_or_path=pretrained_vit,
                torch_dtype=torch.float16,
            ),
        )

        projector_path = osp.join(pretrained_pth, "projector")
        self.model.projector = AutoModel.from_pretrained(projector_path, torch_dtype=torch.float16, trust_remote_code=True)

        self._image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # use standard CLIP processor
            trust_remote_code=True,
            size=resolution,
            crop_size=resolution,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_llm,
            trust_remote_code=True,
            padding_side="right",
        )
        # compute token merge ratio settings
        self.patch_size = self._model.visual_encoder.config.patch_size
        self.num_layers = self._model.visual_encoder.config.num_hidden_layers
        self.token_merge_ratio = token_merge_ratio

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
                self._model.visual_encoder = accelerator.prepare(self.model.visual_encoder)
                self._model.projector = accelerator.prepare(self.model.projector)
            else:  # DistributedType.MULTI_GPU
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
                self._model.visual_encoder = accelerator.prepare_model(self.model.visual_encoder, evaluation_mode=True)
                self._model.projector = accelerator.prepare_model(self.model.projector, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

        # For Video Caption
        self.video_decode_backend = video_decode_backend
        self.max_frames_num = int(max_frames_num)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def process_images(self, images, image_processor, model_cfg):
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        elif image_aspect_ratio == "anyres":
            for image in images:
                image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if visuals:
                image = self.process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0]

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts[0]
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                data = dict()
                data["pixel_values"] = image_tensor
                data["input_ids"] = input_ids
                data["attention_mask"] = attention_masks
                self.model.visual_encoder.reset_tome_r(self.token_merge_ratio)
                output = self.model(data, mode="tensor")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def extract_keyframes(self, video_path, keyframes):
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate
        time_base = video_stream.time_base
        frames = []

        for keyframe in keyframes:
            keyframe_time = float(keyframe)
            frame_number = int(keyframe_time * fps)
            container.seek(int(keyframe_time / time_base))
            found = False
            for packet in container.demux(video=0):
                for frame in packet.decode():
                    if frame.index >= frame_number:
                        frames.append(frame)
                        found = True
                        break
                if found:
                    break

            if not found:
                container.seek(-1, any_frame=False)
                for packet in container.demux(video=0):
                    for frame in packet.decode():
                        pass
                frames.append(frame)

        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # the length of visuals is 1, equal to batchsize
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if visuals:
                if isinstance(visuals[0], dict):
                    video_path = visuals[0]["video_path"]
                    keyframe = visuals[0]["keyframe"]
                    video = self.extract_keyframes(video_path, keyframe)
                    image_tensor = self.process_images(video, self._image_processor, self._config).cuda()
                elif isinstance(visuals, list):
                    print(visuals[0])
                    if isinstance(visuals[0], Image.Image):
                        image_tensor = self.process_images(visuals, self._image_processor, self._config)
                    else:
                        if visuals[0].endswith("mp4"):
                            if self.video_decode_backend == "decord":
                                video = self.load_video(visuals[0], self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                video = read_video_pyav(visuals[0], num_frm=self.max_frames_num)
                            image_tensor = self.process_images(video, self._image_processor, self._config).cuda()
                        elif visuals[0].endswith("mkv"):
                            assert self.video_decode_backend == "pyav", "we only tested this case, decord may not work"
                            video = read_video_pyav(visuals[0], num_frm=self.max_frames_num)
                            image_tensor = self.process_images(video, self._image_processor, self._config).cuda()

                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

            else:
                image_tensor = None

            question_input = []

            for visual, context in zip(visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    if isinstance(visuals[0], dict):
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video)
                    elif isinstance(visuals, list):
                        if isinstance(visuals[0], Image.Image):
                            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                        else:
                            if visual.endswith("mp4") or visual.endswith("mkv"):
                                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video)

                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context

                else:
                    question = context
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # The above for loop has bugs. When there is no visuals, e.g. pure text,
            # there will be no for loop execute resulting in an empty question_input (because no visuals)
            # Scenario 1 won't even be execute
            if len(visuals) == 0:
                for context in contexts:
                    question = context
                    conv = conv_templates[self.conv_template].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # preconfigure gen_kwargs with defaults
            if isinstance(visuals[0], dict):
                gen_kwargs["image_sizes"] = [video[idx].size for idx in range(len(video))]
            elif isinstance(visuals, list):
                if isinstance(visuals[0], Image.Image):
                    gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
                else:
                    if visuals[0].endswith("mp4"):
                        gen_kwargs["image_sizes"] = [video[idx].size for idx in range(len(video))]

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            try:
                data = dict()
                if isinstance(visuals[0], dict):
                    data["pixel_values"] = image_tensor.unsqueeze(0)
                elif isinstance(visuals, list):
                    if isinstance(visuals[0], Image.Image):
                        data["pixel_values"] = image_tensor
                    else:
                        if visuals[0].endswith("mp4") or visuals[0].endswith("mkv"):
                            data["pixel_values"] = image_tensor.unsqueeze(0)

                data["input_ids"] = input_ids
                data["attention_mask"] = attention_masks
                self.model.visual_encoder.reset_tome_r(self.token_merge_ratio)
                output = self.model(data, mode="inference")
                cont = self.model.llm.generate(
                    **output,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            print(text_outputs)

            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
