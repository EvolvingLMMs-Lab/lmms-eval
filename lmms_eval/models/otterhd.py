from transformers import FuyuForCausalLM, AutoTokenizer, FuyuImageProcessor, FuyuProcessor
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import torch
from PIL import Image
from typing import List, Optional, Union, Tuple
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from tqdm import tqdm


@register_model("otterhd")
class OtterHD(lmms):
    """
    OtterHD Model
    """

    def __init__(
        self,
        pretrained: str = "Otter-AI/OtterHD-8B",
        resolution: str = "360x360",
        device: Optional[str] = "cuda",
        max_new_tokens: int = 256,
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = FuyuForCausalLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        height, width = map(int, resolution.split("x"))
        self.image_processor = FuyuImageProcessor(size={"height": height, "width": width})
        self.processor = FuyuProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "OtterHD currently does not support batched generation."

    @property
    def max_length(self):
        # Assuming max_length is the sum of max context tokens and max new tokens
        return self.tokenizer.model_max_length

    # @property
    # def max_gen_toks(self) -> int:
    #     return self.max_new_tokens

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

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
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, visuals = zip(*chunk)
            visuals = self.flatten(visuals)
            gen_kwargs = all_gen_kwargs[0]

            if isinstance(visuals, list):
                visuals = [visuals[0]]

            formatted_contexts = f"User: {contexts[0]} Assistant:"
            model_inputs = self.processor(text=[formatted_contexts], images=visuals, device=self.device)
            for k, v in model_inputs.items():
                model_inputs[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(self.device, non_blocking=True) for vv in v]

            for index in range(len(model_inputs["image_patches"])):
                model_inputs["image_patches"][index] = model_inputs["image_patches"][index].to(dtype=next(self.model.parameters()).dtype)

            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
            generation_output = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
            response = generation_text[0].split("\x04")[1].strip(" ").strip("\n")
            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for llava yet"

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        # TODO
        assert False, "We have not implemented this function for llava yet"

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
