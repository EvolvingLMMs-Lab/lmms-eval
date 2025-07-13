from typing import List, Tuple

from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.vllm import VLLM as VLLMSimple
from lmms_eval.protocol import ChatMessages

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None


@register_model("vllm_chat")
class VLLM(VLLMSimple):
    is_simple = False

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            batched_messages = []
            for idx in range(len(batch_requests)):
                ctx, doc_to_messages, gen_kwargs, doc_id, task, split = batch_requests[idx].arguments
                chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
                chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95

                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                }
                sampling_params = SamplingParams(**params)
                messages = chat_messages.to_openai_messages()

                batched_messages.append(messages)

            sampling_params = SamplingParams(**params)

            if self.chat_template is not None:
                with open(self.chat_template, "r") as f:
                    chat_template = f.read()
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=chat_template)
            else:
                breakpoint()
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages)
            response_text = [o.outputs[0].text for o in response]

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
