"""baichuan_omni_interleave — Baichuan-Omni over interleaved doc_to_messages.

See `_interleave_base.InterleaveChatMixin` for the shared rationale and
request loop. Only the Baichuan-Omni-specific inference step lives here.

Baichuan-Omni's prompt is a single string with media encoded as
`<start>{"local"|"path": ...}<end>` segments. We emit those segments in the
exact order of the doc_to_messages blocks so the question stem and every
option keep their positions.
"""

import torch
import ujson
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._interleave_base import InterleaveChatMixin
from lmms_eval.models.simple.baichuan_omni import BaichuanOmni


@register_model("baichuan_omni_interleave")
class BaichuanOmniInterleave(InterleaveChatMixin, BaichuanOmni):
    """Baichuan-Omni that consumes interleaved doc_to_messages prompts."""

    def _interleaved_content(self, messages: list) -> str:
        parts = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                t = c.get("type")
                if t == "text":
                    parts.append(c.get("text", ""))
                elif t == "image":
                    parts.append(self.image_start_token + ujson.dumps({"local": c["url"]}, ensure_ascii=False) + self.image_end_token)
                elif t == "video":
                    parts.append(self.video_start_token + ujson.dumps({"local": c["url"]}, ensure_ascii=False) + self.video_end_token)
                elif t == "audio":
                    parts.append(self.audio_start_token + ujson.dumps({"path": c["url"]}, ensure_ascii=False) + self.audio_end_token)
        return "".join(parts)

    def _infer_one(self, messages: list, gen_kwargs: dict) -> str:
        user_content = self._interleaved_content(messages)
        prompt = self._format_prompt(user_content)

        inputs = self.model.processor([prompt])
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda() if inputs.attention_mask is not None else None
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tokenizer": self.tokenizer,
        }
        if inputs.audios is not None:
            model_inputs["audios"] = inputs.audios.cuda()
        if inputs.encoder_length is not None:
            model_inputs["encoder_length"] = inputs.encoder_length.cuda()
        if inputs.bridge_length is not None:
            model_inputs["bridge_length"] = inputs.bridge_length.cuda()
        if inputs.images is not None:
            model_inputs["images"] = [torch.tensor(img, dtype=torch.float32).cuda() for img in inputs.images]
            if inputs.patch_nums is not None:
                model_inputs["patch_nums"] = inputs.patch_nums
            if inputs.images_grid is not None:
                model_inputs["images_grid"] = inputs.images_grid
        if inputs.videos is not None:
            model_inputs["videos"] = [torch.tensor(vid, dtype=torch.float32).cuda() for vid in inputs.videos]
            if inputs.videos_patch_nums is not None:
                model_inputs["videos_patch_nums"] = inputs.videos_patch_nums
            if inputs.videos_grid is not None:
                model_inputs["videos_grid"] = inputs.videos_grid

        max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
        temperature = gen_kwargs.get("temperature", 0.0)
        do_sample = temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=["<|endoftext|>"],
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                use_cache=self.use_cache,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
        generated_ids = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
