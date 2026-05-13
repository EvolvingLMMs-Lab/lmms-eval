import time
from typing import List

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.phi4_multimodal import Phi4 as Phi4Simple
from lmms_eval.protocol import ChatMessages


@register_model("phi4_multimodal")
class Phi4(Phi4Simple):
    is_simple = False

    def get_role_tag(self, role: str):
        if role == "system":
            return "<|system|>"
        elif role == "user":
            return "<|user|>"
        elif role == "assistant":
            return "<|assistant|>"
        else:
            return f"<|{role}|>"

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            chat_messages = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids in doc_id]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            audios = []
            for messages in chat_messages:
                visual, video, audio = messages.extract_media()
                visuals.extend(visual)
                videos.extend(video)
                audios.extend(audio)

                gen_kwargs = all_gen_kwargs[0]

                prompt_parts = []
                images_list = []
                image_counter = 1

                for msg in messages.messages:
                    role_tag = self.get_role_tag(msg.role)
                    prompt_parts.append(role_tag)
                    for content in msg.content:
                        if content.type == "image":
                            prompt_parts.append(f"<|image_{image_counter}|>")
                            if isinstance(content.url, str):
                                img = Image.open(content.url)
                            else:
                                img = content.url
                            images_list.append(img)
                            image_counter += 1
                        elif content.type == "video":
                            frames = self.load_video(content.url, self.max_frame_num)
                            for frame in frames:
                                prompt_parts.append(f"<|image_{image_counter}|>")
                                images_list.append(Image.fromarray(np.uint8(frame)))
                                image_counter += 1
                        elif content.type == "audio":
                            pass
                        elif content.type == "text":
                            prompt_parts.append(content.text)
                    prompt_parts.append("<|end|>")

                    prompt_text = "".join(prompt_parts) + "<|assistant|>"

                    text, images, processed_audios = self.default_process(visuals + videos + audios, [""])
                    text = prompt_text

                    inputs = self._processor(text=text, images=images, audios=processed_audios, return_tensors="pt").to(self.device)

                    if self.accelerator.is_main_process:
                        eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

                    if "max_new_tokens" not in gen_kwargs:
                        gen_kwargs["max_new_tokens"] = 1024
                    if "temperature" not in gen_kwargs:
                        gen_kwargs["temperature"] = 0
                    if "top_p" not in gen_kwargs:
                        gen_kwargs["top_p"] = None
                    if "num_beams" not in gen_kwargs:
                        gen_kwargs["num_beams"] = 1

                    start_time = time.time()
                    try:
                        cont = self.model.generate(
                            **inputs,
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                            use_cache=self.use_cache,
                            pad_token_id=self.eot_token_id,
                            num_logits_to_keep=0,
                        )
                        end_time = time.time()
                    except Exception as e:
                        eval_logger.error(f"Error generating text: {e}")
                        cont = inputs["input_ids"]
                        end_time = time.time()
                    cont = cont[:, inputs["input_ids"].shape[-1] :]
                    text_outputs = self._processor.batch_decode(cont, skip_special_tokens=True)[0]

                    total_elapsed_time += end_time - start_time
                    total_tokens += len(cont[0])

                    if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                        eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

                    token_counts = TokenCounts(output_tokens=len(cont[0])) if cont is not None else None
                    res.append(GenerationResult(text=text_outputs, token_counts=token_counts))
                    self.cache_hook.add_partial("generate_until", (text, gen_kwargs), text_outputs)
                    pbar.update(1)
        res = re_ords.get_original(res)

        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
