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

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            gen_kwargs = dict(all_gen_kwargs[0])

            doc = self.task_dict[task_name][split_name][doc_id[0]]
            raw_messages = doc_to_messages[0](doc)
            chat_messages = ChatMessages(**{"messages": raw_messages})

            try:
                answer = self._infer_one(chat_messages, gen_kwargs)
            except Exception as e:
                eval_logger.error(f"Error in generating: {e}\n{traceback.format_exc()}")
                answer = ""

            res.append(answer)
            self.cache_hook.add_partial("generate_until", (ctx[0], gen_kwargs), answer)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
