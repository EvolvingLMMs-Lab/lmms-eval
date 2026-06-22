"""baichuan_omni — Baichuan-Omni chat-style wrapper.

See `_chat_base.ChatMixin` for the shared rationale and request loop.
Only the Baichuan-Omni-specific inference step lives here.

Baichuan-Omni's prompt is a single string with media encoded as
`<start>{"local"|"path": ...}<end>` segments. We emit those segments in the
exact order of the chat blocks so the question stem and every option keep
their positions.
"""

import torch
import ujson

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.models.simple.baichuan_omni import BaichuanOmni as BaichuanOmniSimple
from lmms_eval.protocol import ChatMessages


@register_model("baichuan_omni_chat")
class BaichuanOmni(ChatMixin, BaichuanOmniSimple):
    """Baichuan-Omni that consumes chat-style doc_to_messages prompts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # XModBench items carry 4 vision options; at Baichuan's default
        # ~1 MP/image (and 768*28*28 video) the a2v/t2v configs OOM even on
        # 4x48GB. OmniImageProcessor caches max_pixels at __init__
        # (processor_omni.py:164), so setting config.max_pixels afterwards is
        # too late — set the cached attribute on the already-built
        # visual/video sub-processors directly. No upstream change.
        proc = getattr(self.model, "processor", None)
        for sub in ("visual_processor", "video_processor"):
            ip = getattr(proc, sub, None)
            if ip is not None and hasattr(ip, "max_pixels"):
                ip.max_pixels = 256 * 28 * 28  # ~0.2 MP (vs ~1 MP / 0.6 MP)

    def _interleaved_content(self, chat_messages: ChatMessages) -> str:
        parts = []
        for msg in chat_messages.messages:
            for c in msg.content:
                if c.type == "text":
                    parts.append(c.text)
                elif c.type == "image":
                    parts.append(self.image_start_token + ujson.dumps({"local": c.url}, ensure_ascii=False) + self.image_end_token)
                elif c.type == "video":
                    parts.append(self.video_start_token + ujson.dumps({"local": c.url}, ensure_ascii=False) + self.video_end_token)
                elif c.type == "audio":
                    parts.append(self.audio_start_token + ujson.dumps({"path": c.url}, ensure_ascii=False) + self.audio_end_token)
        return "".join(parts)

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        user_content = self._interleaved_content(chat_messages)
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
        generated_ids = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
