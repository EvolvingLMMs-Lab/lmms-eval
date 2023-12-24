import torch
import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor

import copy
from collections import defaultdict
from tqdm import tqdm
from lmm_eval import utils
from lmm_eval.api.instance import Instance
from lmm_eval.api.model import LMM
from lmm_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from typing import List, Optional, Union, Tuple
eval_logger = utils.eval_logger
from lmm_eval.utils import stop_sequences_criteria



@register_model("llava")
class Llava(LMM):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        truncation: Optional[bool] = True,
        max_length: Optional[int] = 4096,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision = None,
        use_flash_attention_2=True,
        **kwargs,
    ) -> None:
        super().__init__()
        model_kwargs = kwargs if kwargs else {}

        accelerator = Accelerator()
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self._model = LlavaForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=utils.get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                revision=revision,
                use_flash_attention_2=use_flash_attention_2,
                **model_kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(pretrained)
        self.model.eval()
        self.model.tie_weights()
        

        self.truncation = truncation
        self._max_length = max_length
        self.batch_size_per_gpu = int(batch_size)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self._device = torch.device(
                f"cuda:{accelerator.local_process_index}"
            )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._device = device
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

        

        
        
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config
    
    @property
    def processor(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._processor
    @property
    def tokenizer(self):
        return self.processor.tokenizer
    
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


    @property
    def max_gen_toks(self) -> int:
        return 256

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

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
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
        # TODO
        assert False, "We have not implemented this function for llava yet"

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        # TODO
        assert False, "We have not implemented this function for llava yet"

    def _model_generate(self, batch_features, max_length, stop, **generation_kwargs):
        # we require users to pass do_sample=True explicitly
        # for non-greedy gen. This should be reevaluated when considering beam search.
        if "do_sample" not in generation_kwargs:
            generation_kwargs["do_sample"] = False
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, batch_features.input_ids.shape[0]
        )
        return self.model.generate(
            **batch_features,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            use_cache=True,
            **generation_kwargs,
        )

    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = defaultdict(list)
        re_ords = {}

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
        grouper = utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))

        for key, re_ord in re_ords.items():
            chunks = utils.chunks(
                re_ord.get_reordered(),
                n=self.batch_size,
                fn=None,
            )
            for chunk in chunks:
                contexts, all_gen_kwargs, visuals = zip(*chunk)
                def flatten(input):
                    new_list = []
                    for i in input:
                        for j in i:
                            new_list.append(j)
                    return new_list
                visuals = flatten(visuals)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [kwargs]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                    )
                if not until:
                    until = [self.tok_decode(self.eot_token_id)]
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks

                max_ctx_len = self.max_length - max_gen_toks


                # encode, pad, and truncate contexts for this batch
                batch_features = self.processor(
                    contexts,
                    visuals,
                    max_length=max_ctx_len,
                    truncation=self.truncation,
                ).to(self.device)


                if "max_length" not in kwargs:
                    kwargs["max_length"] = batch_features.input_ids.shape[1] + max_gen_toks

                # perform batched generation
                cont = self._model_generate(
                    batch_features=batch_features,
                    stop=until,
                    **kwargs,
                )

                cont_toks_list = cont.tolist()
                for cont_toks, context in zip(cont_toks_list, contexts):
                    # discard context + left-padding toks if using causal decoder-only LM
                    cont_toks = cont_toks[batch_features.input_ids.shape[1] :]
                    s = self.tokenizer.decode(cont_toks, skip_special_tokens=True)

                    # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                    for term in until:
                        if len(term) > 0:
                            # ignore '' separator,
                            # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                            s = s.split(term)[0]

                    res[key].append(s)

                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), s
                    )
                    pbar.update(1)
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()
        return grouper.get_original(res)
