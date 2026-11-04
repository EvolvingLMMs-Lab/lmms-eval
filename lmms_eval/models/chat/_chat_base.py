"""Shared scaffolding for chat-style omni model wrappers.

Many lmms-eval *simple* omni model wrappers (qwen2.5-omni, qwen3-omni,
omnivinci, baichuan-omni, minicpm-o, vita, ...) only attach one media object
per request via `doc_to_visual`. Tasks like XModBench put media in the
question stem AND every answer option (up to 5 per item), so the simple path
drops most of it.

The fix is the same for every model: become a chat-style model
(`is_simple = False`), read the task's `doc_to_messages` output, and feed the
full prompt to the underlying model. Only the final
"messages -> model output string" step is model-specific.

`ChatMixin` implements the common request loop once. A concrete wrapper
subclasses it (plus the model's simple class) and implements a single method:

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str: ...

Per-media size/frame caps that match the upstream XModBench/AudioBench
runners are exposed as module constants so wrappers stay consistent.
"""

import traceback
from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages

# Per-media caps. The upstream AudioBench runner used fps=12/max_frames=60/
# 512px, but that assumes ~80 GB GPUs; on 24 GB a5000s a single
# video-condition item plus 4 audio options OOMs (~50% of v2a/v2t video
# samples were silently dropped). A tighter video budget keeps every sample
# on-GPU at minor frame-density cost (XModBench video tasks — emotion,
# spatial, temporal — don't need 60 frames).
VIDEO_KWARGS = {"fps": 2, "max_frames": 16, "max_pixels": 384 * 384}
IMAGE_KWARGS = {"max_pixels": 512 * 512}


class ChatMixin:
    """Mixin providing a chat-style generate_until over doc_to_messages.

    Subclasses may override `video_kwargs` / `image_kwargs` class attributes
    to use a tighter media budget (e.g. Baichuan-Omni's video path needs far
    more memory than Qwen-Omni's for the same clip).

    Subclasses must define `_infer_one(self, chat_messages, gen_kwargs) -> str`.
    Subclasses may opt into native text-output batching by defining
    `_infer_batch(self, conversations, gen_kwargs) -> list[str]`.
    Other subclasses continue to infer each request individually.
    """

    is_simple = False

    # Per-model media budget; override in a subclass if it OOMs.
    video_kwargs = VIDEO_KWARGS
    image_kwargs = IMAGE_KWARGS

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        try:
            for chunk in chunks:
                infer_batch = getattr(self, "_infer_batch", None)
                if callable(infer_batch):
                    conversations = []
                    cache_keys = []
                    for ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split in chunk:
                        gen_kwargs = dict(all_gen_kwargs)
                        gen_kwargs.pop("raise_on_error", False)
                        doc = self.task_dict[task][split][doc_id]
                        raw_messages = doc_to_messages(doc)
                        conversations.append(ChatMessages(**{"messages": raw_messages}))
                        cache_keys.append((ctx, gen_kwargs))

                    # Collator groups equal generation settings. Give inference
                    # its own copy so defaults cannot enter any cache key.
                    batch_kwargs = dict(cache_keys[0][1])
                    raise_on_error = chunk[0][2].get("raise_on_error", False)
                    try:
                        answers = infer_batch(conversations, batch_kwargs)
                    except Exception as e:
                        if raise_on_error:
                            raise
                        eval_logger.error(f"Error in generating: {e}\n{traceback.format_exc()}")
                        answers = [""] * len(chunk)

                    # A broken adapter result is a contract error, not a model
                    # inference failure. Validate before cache/progress writes.
                    if not isinstance(answers, (list, tuple)) or len(answers) != len(chunk) or any(not isinstance(answer, str) for answer in answers):
                        raise ValueError("_infer_batch must return one string per request")
                    for key, answer in zip(cache_keys, answers):
                        res.append(answer)
                        self.cache_hook.add_partial("generate_until", key, answer)
                        pbar.update(1)
                    continue

                for ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split in chunk:
                    gen_kwargs = dict(all_gen_kwargs)
                    raise_on_error = gen_kwargs.pop("raise_on_error", False)

                    doc = self.task_dict[task][split][doc_id]
                    raw_messages = doc_to_messages(doc)
                    chat_messages = ChatMessages(**{"messages": raw_messages})

                    try:
                        answer = self._infer_one(chat_messages, gen_kwargs)
                    except Exception as e:
                        if raise_on_error:
                            raise
                        eval_logger.error(f"Error in generating: {e}\n{traceback.format_exc()}")
                        answer = ""

                    res.append(answer)
                    self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), answer)
                    pbar.update(1)

            return re_ords.get_original(res)
        finally:
            pbar.close()
