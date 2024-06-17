import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from accelerate import Accelerator, DistributedType
from transformers import FuyuForCausalLM, AutoTokenizer, FuyuImageProcessor, FuyuProcessor
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import torch
from PIL import Image
from typing import List, Optional, Union, Tuple
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
import logging

import logging

eval_logger = logging.getLogger("lmms-eval")

eval_logger = logging.getLogger("lmms-eval")


@register_model("fuyu")
class Fuyu(lmms):
    """
    Fuyu Model
    """

    def __init__(
        self,
        pretrained: str = "adept/fuyu-8b",
        device: Optional[str] = "cuda",
        max_new_tokens: int = 256,
        batch_size: Optional[Union[int, str]] = 1,
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

        self._model = FuyuForCausalLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model.eval()
        self.model.tie_weights()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = self.model.config

        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.batch_size_per_gpu = int(batch_size)
        accelerator = Accelerator()
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
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

        '''if accelerator.num_processes > 1:
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
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes'''

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
        # Assuming max_length is the sum of max context tokens and max new tokens
        return self.tokenizer.model_max_length

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

    def flatten(self, input, only_get_first=False):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
                if only_get_first:
                    break
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
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals, only_get_first=True)
            gen_kwargs = all_gen_kwargs[0]

            # if isinstance(visuals[0], list):
            #     visuals = [visuals[idx][0] for idx in range(len(visuals))]  # get the first image in multi-image scenarios.

            # assert len(contexts) == self.batch_size_per_gpu, f"Expected contexts batch size {self.batch_size_per_gpu}, got {len(contexts)}"
            # assert len(visuals) == self.batch_size_per_gpu, f"Expected visuals batch size {self.batch_size_per_gpu}, got {len(visuals)}"
            formatted_contexts = [f"{context}\n" for context in contexts]
            model_inputs = self.processor(text=formatted_contexts, images=visuals, device=self.device)
            for k, v in model_inputs.items():
                model_inputs[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(self.device, non_blocking=True) for vv in v]

            for index in range(len(model_inputs["image_patches"])):
                model_inputs["image_patches"][index] = model_inputs["image_patches"][index].to(dtype=next(self.model.parameters()).dtype)

            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            # generation_output = self.model.generate(
            #     **model_inputs, temperature=gen_kwargs["temperature"], max_new_tokens=gen_kwargs["max_new_tokens"], top_p=gen_kwargs["top_p"], num_beams=gen_kwargs["num_beams"], pad_token_id=self.tokenizer.eos_token_id
            # )
            generation_output = self.model.generate(**model_inputs, max_new_tokens=gen_kwargs["max_new_tokens"])
            generation_texts = self.processor.batch_decode(generation_output, skip_special_tokens=True)
            response = [gen_text.split("\x04")[1].strip(" ").strip("\n") for gen_text in generation_texts]
            res.extend(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
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
            formatted_contexts = [f"{contexts}\n"]
            formatted_continuation = [f"{contexts}\n{continuation}"]
            model_inputs = self.processor(text=formatted_continuation, images=visuals, device=self.device)
            for k, v in model_inputs.items():
                model_inputs[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(self.device, non_blocking=True) for vv in v]

            for index in range(len(model_inputs["image_patches"])):
                model_inputs["image_patches"][index] = model_inputs["image_patches"][index].to(dtype=next(self.model.parameters()).dtype)

            labels = model_inputs["input_ids"].clone()
            contxt_id = self.processor(text=formatted_contexts, return_tensors="pt")["input_ids"]
            labels[: len(contxt_id)] = -100
            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : model_inputs["input_ids"].shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

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
