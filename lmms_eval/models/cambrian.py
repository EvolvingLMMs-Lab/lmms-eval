import os
import uuid
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from PIL import Image
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from cambrian.conversation import conv_templates
    from cambrian.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from cambrian.model.builder import load_pretrained_model
except ImportError:
    eval_logger.error("Cambrian is not installed. Please install it by running `pip install cambrian`.")

# Model Constants
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def process(image, question, tokenizer, image_processor, model_config, conv_mode):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
    else:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(role, allowed_special=set(tokenizer.IMAGE_ST)) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += nl_tokens + im_start_tokens + _tokenize_str("user", query)[1] + im_end_tokens + nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


@register_model("cambrian")
class Cambrian(lmms):
    def __init__(
        self,
        pretrained: str = "nyu-visionx/cambrian-8b",
        device: Optional[str] = "cuda",
        device_map="auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert not kwargs, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}") if accelerator.num_processes > 1 else device

        self.model_name = get_model_name_from_path(pretrained)
        tokenizer, model, self.image_processor, context_len = load_pretrained_model(pretrained, None, self.model_name, device_map=self._device)

        self.conv_mode = {"cambrian-8b": "llama_3", "cambrian-13b": "vicuna_v1", "cambrian-34b": "chatml_direct"}.get(self.model_name)

        if not self.conv_mode:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self._model = model
        self._tokenizer = tokenizer
        self._model.eval()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._rank = 0
        self._world_size = 1

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], "Unsupported distributed type. Only DDP and FSDP are supported."
            self._model = accelerator.prepare(self.model) if accelerator.distributed_type == DistributedType.FSDP else accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)

        self.accelerator = accelerator

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            continuation = doc_to_target if isinstance(doc_to_target, str) else doc_to_target(self.task_dict[task][split][doc_id])
            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])

            query = []
            visual_paths = []
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual_path = f"/tmp/{name}.png"
                visual.save(visual_path)
                visual_paths.append(visual_path)
                query.append({"image": visual_path})

            context_query = query.copy()
            context_query.append({"text": contexts})
            query.append({"text": contexts + continuation})

            context_query = self.tokenizer.from_list_format(context_query)
            query = self.tokenizer.from_list_format(query)

            _, context_tokens = make_context(
                self.tokenizer, context_query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format
            )
            context_tokens = torch.tensor([context_tokens])

            _, continuation_tokens = make_context(self.tokenizer, query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format)
            continuation_tokens = torch.tensor([continuation_tokens]).to(self.model.device)
            attn_mask = torch.ones_like(continuation_tokens).to(self.model.device)
            labels = continuation_tokens.clone().to(self.model.device)
            labels[:, : context_tokens.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=continuation_tokens, labels=labels, attention_mask=attn_mask)

            loss = outputs.loss
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = continuation_tokens[:, context_tokens.shape[1] :]
            greedy_tokens = greedy_tokens[:, context_tokens.shape[1] : continuation_tokens.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    @staticmethod
    def flatten(input_list):
        return [item for sublist in input_list for item in sublist]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = self.flatten([doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id])

            visual_paths = []
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual_path = f"/xpfs/public/gezhang/zk/lmms-eval/lmms_eval/tmp/{name}.png"
                visual.save(visual_path)
                visual_paths.append(visual_path)

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            gen_kwargs.setdefault("image_sizes", [visuals[0].size] if visuals else None)
            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)

            until.append("<|eot_id|>")

            image = Image.open(visual_paths[0]).convert("RGB")
            question = contexts[0]

            input_ids, image_tensor, image_sizes, prompt = process(image, question, self.tokenizer, self.image_processor, self.model.config, self.conv_mode)
            input_ids = input_ids.to(device=self.model.device, non_blocking=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=gen_kwargs["temperature"] > 0,
                    temperature=gen_kwargs["temperature"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=True,
                )

            text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            for term in until:
                if term:
                    text_outputs = text_outputs.split(term)[0]

            print(text_outputs)
            res.append(text_outputs)

            for visual_path in visual_paths:
                try:
                    os.remove(visual_path)
                except OSError:
                    pass

            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
