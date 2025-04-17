from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from apps.plm.generate import (PackedCausalTransformerGenerator,
                               PackedCausalTransformerGeneratorArgs,
                               load_consolidated_model_and_tokenizer)
from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform
from core.transforms.video_transform import get_video_transform


@register_model("plm")
class PerceptionLM(lmms):
    """
    Perception Lanugate Model (PLM)
    "Paste the paper link"
    "Paste the github link"
    "Paste the huggingface link"
    """

    def __init__(
        self,
        pretrained: str = "Replace/with/huggingface/model/link",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        compile_prefilling=False,
        reduce_generation_overhead=False,
        max_tokens=11264,
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")

        # Collect all arguments into a dictionary
        args = {
            "pretrained": pretrained,
            "device": device,
            "batch_size": batch_size,
            "compile_prefilling": compile_prefilling,
            "reduce_generation_overhead": reduce_generation_overhead,
            "max_tokens": max_tokens,
            **kwargs,  # Include any additional keyword arguments
        }
        # Convert the dictionary to a dotlist format
        dotlist = [f"{key}={value}" for key, value in args.items()]
        cfg = OmegaConf.from_dotlist(dotlist)
        gen_cfg = dataclass_from_dict(PackedCausalTransformerGeneratorArgs, cfg, strict=False)
        # Load PLM model
        eval_logger.info(f"Lodding PLM model from {cfg.pretrained}")
        model, tokenizer, config = load_consolidated_model_and_tokenizer(cfg.pretrained)

        # Create preprocessors (transforms)
        processor = {}
        vision_input_type = config.get("model").get("vision_input_type", "thumb+tile")
        max_num_tiles = config.get("model").get("max_num_tiles", 36)
        processor["image"] = get_image_transform(vision_input_type=vision_input_type, image_res=model.vision_model.image_size, max_num_tiles=max_num_tiles)
        processor["video"] = get_video_transform(image_res=model.vision_model.image_size)
        self._video_max_frames = config.get("model").get("video_max_frames", 32)

        # Create PLM generator
        eval_logger.info(f"Creating packed generator with gen_cfg: {gen_cfg}")
        generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

        # Set the class variables
        self._tokenizer = tokenizer
        self._processor = processor
        self._model = model
        self._generator = generator
        self.batch_size_per_gpu = int(batch_size)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def generator(self):
        return self._generator

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of text is more accurate for what we're doing than end of sentence
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def video_max_frames(self):
        return self._video_max_frames

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for PLM")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0], add_bos=False, add_eos=False)
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            messages = []
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        video_info = (visual, self.video_max_frames, None, None, None)
                        visual, _ = self.processor["video"](video_info)
                        message = (context, visual)
                    elif isinstance(visual, Image.Image):  # Single image
                        visual = visual.convert("RGB")
                        visual, _ = self.processor["image"](visual)
                        message = (context, visual)
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images or Video Frames
                        visual = [image.convert("RGB") for image in visual]
                        visual, _ = self.processor["video"]._process_multiple_images_pil(visual)
                        message = (context, visual)
                    else:
                        # Text-only sample
                        raise NotImplementedError("Text-only input is not yet supported.")
                else:
                    # Text-only sample
                    raise NotImplementedError("Text-only input is not yet supported.")

                messages.append(message)

            gen_kwargs = all_gen_kwargs[0]
            if "max_new_tokens" in gen_kwargs:
                self.generator.max_gen_len = gen_kwargs["max_new_tokens"]
            if "temperature" in gen_kwargs:
                self.generator.temperature = gen_kwargs["temperature"]
            # Default for PLM
            self.generator.top_p = None
            self.generator.top_k = 100

            generation, loglikelihood, greedy = self.generator.generate(messages)

            for gen, context in zip(generation, contexts):
                if gen.endswith("."):
                    gen = gen[:-1]
                res.append(gen)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), gen)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented yet.")
