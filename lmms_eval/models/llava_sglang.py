import torch

torch.backends.cuda.matmul.allow_tf32 = True

import logging
from tqdm import tqdm
from datetime import timedelta

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from accelerate import Accelerator, InitProcessGroupKwargs
from typing import List, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

eval_logger = logging.getLogger("lmms-eval")

try:
    import sglang as sgl
    from sglang.lang.chat_template import get_chat_template
except ImportError:
    eval_logger.debug("SGLang is not installed. If you want to use llava_sglang, please install it using pip install 'sglang[all]' ")

if torch.__version__ > "2.1.2":
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_sglang")
class LlavaSglang(lmms):
    """
    Llava Sglang Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        tokenizer: str = "llava-hf/llava-1.5-7b-hf",
        tp_size: int = 1,
        parallel: Optional[Union[int, str]] = 64,
        conv_template="vicuna_v1.1",
        **kwargs,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.tokenizer = tokenizer
        self.tp_size = tp_size
        self.conv_template = conv_template
        torch.multiprocessing.set_start_method("spawn")

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        assert accelerator.num_processes == 1, "Llava-sglang does not support multi-processes yet (it does support tensor parallelism)."
        self._rank = 0
        self._world_size = 1
        self.parallel = parallel

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Llava-sglang does not support loglikelihood evaluation yet")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        runtime = sgl.Runtime(model_path=self.pretrained, tokenizer_path=self.tokenizer, tp_size=self.tp_size)
        runtime.endpoint.chat_template = get_chat_template(self.conv_template)
        sgl.set_default_backend(runtime)

        @sgl.function
        def image_qa(s, image_file, question):
            s += sgl.user(sgl.image(image_file) + question)
            s += sgl.assistant(sgl.gen("answer"))

        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = x[0].split(" ")
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.parallel, batch_fn=None)
        num_iters = len(requests) // self.parallel if len(requests) % self.parallel == 0 else len(requests) // self.parallel + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_id, tasks, splits = zip(*chunk)
            batched_visuals = [doc_to_visual(self.task_dict[task][split][ids]) for ids, task, split, doc_to_visual in zip(doc_id, tasks, splits, doc_to_visuals)]  # [B, N]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 1.0
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if gen_kwargs["top_p"] == 0.0:
                gen_kwargs["top_p"] = 1.0
                gen_kwargs["temperature"] = 0.0
            assert gen_kwargs["num_beams"] == 1

            def save_image_to_temp_file(image):
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=True)
                image.save(temp_file.name)
                return temp_file

            def prepare_arguments_parallel(contexts, batched_visuals, max_workers=64):
                arguments = [None] * len(contexts)  # Initialize with placeholders
                tmp_files = [None] * len(contexts)  # Initialize with placeholders

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Associate each future with its index and content
                    future_to_info = {executor.submit(save_image_to_temp_file, pil_list[0]): (index, context, pil_list) for index, (context, pil_list) in enumerate(zip(contexts, batched_visuals))}

                    for future in as_completed(future_to_info):
                        index, context, pil_list = future_to_info[future]
                        if len(pil_list) > 1:
                            eval_logger.warning("Llava-sglang only supports one visual input per question. Using the first visual input.")
                        try:
                            temp_file = future.result()
                            arguments[index] = {
                                "image_file": temp_file.name,
                                "question": context,
                            }
                            tmp_files[index] = temp_file
                        except Exception as exc:
                            print(f"Generated an exception: {exc}")

                # Filter out any None values in case of exceptions
                arguments = [arg for arg in arguments if arg is not None]
                tmp_files = [tmp_file for tmp_file in tmp_files if tmp_file is not None]

                return arguments, tmp_files

            arguments, tmp_files = prepare_arguments_parallel(contexts, batched_visuals, self.parallel)
            states = image_qa.run_batch(arguments, temperature=gen_kwargs["temperature"], max_new_tokens=gen_kwargs["max_new_tokens"], top_p=gen_kwargs["top_p"], num_threads=self.parallel, progress_bar=False)

            text_outputs = [state["answer"].strip() for state in states]
            # clean up the temporary files
            for tmp_file in tmp_files:
                tmp_file.close()
            res.extend(text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        runtime.shutdown()
        return res
