"""omnivinci — OmniVinci chat-style wrapper.

See `_chat_base.ChatMixin` for the shared rationale and request loop.
Only the OmniVinci/VILA-specific inference step lives here.
"""

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.models.simple.omnivinci import OmniVinci as OmniVinciSimple
from lmms_eval.protocol import ChatMessages


@register_model("omnivinci_chat")
class OmniVinci(ChatMixin, OmniVinciSimple):
    """OmniVinci that consumes chat-style doc_to_messages prompts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The upstream omnivinci wrapper only sets processor.config
        # .load_audio_in_video. The official AudioBench OmniVinci runner also
        # sets audio_chunk_length and num_video_frames; without
        # audio_chunk_length the processor builds an incomplete
        # mm_info["audio_info"] and __embed_media_tokens raises IndexError on
        # the "sound" branch. Mirror the official settings here (no change to
        # the upstream model file).
        cfg = getattr(self.processor, "config", None)
        if cfg is not None:
            cfg.audio_chunk_length = "max_3600"
            cfg.num_video_frames = getattr(self, "num_video_frames", 128)
            cfg.load_audio_in_video = getattr(self, "load_audio_in_video", True)
        mcfg = getattr(self._model, "config", None)
        if mcfg is not None:
            mcfg.audio_chunk_length = "max_3600"
        # Note: a2v/t2v (4 vision options) and t2a/v2a (4 audio options) hit
        # VILA-internal limits under interleaved prompts — dynamic_s2 tiling
        # OOMs, image_aspect_ratio=resize degenerates to empty output, and
        # multi-audio raises IndexError in __embed_media_tokens. These are
        # not resolvable without upstream model edits, so OmniVinci is
        # reported best-effort on its 2 clean configs (a2t, v2t).

    def _build_message(self, chat_messages: ChatMessages) -> list:
        """Chat blocks -> OmniVinci message.

        OmniVinci/VILA's mm_info builder indexes media by *sample*; an extra
        system turn shifts that indexing and triggers an IndexError in
        __embed_media_tokens. The upstream AudioBench OmniVinci runner uses a
        single user message with no system role — mirror it exactly.
        """
        user_content = []
        for msg in chat_messages.messages:
            for c in msg.content:
                if c.type == "text":
                    user_content.append({"type": "text", "text": c.text})
                elif c.type == "image":
                    user_content.append({"type": "image", "image": c.url})
                elif c.type == "audio":
                    user_content.append({"type": "audio", "audio": c.url})
                elif c.type == "video":
                    user_content.append({"type": "video", "video": c.url})
        return [{"role": "user", "content": user_content}]

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        message = self._build_message(chat_messages)

        vila_text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        inputs = self.processor([vila_text])
        if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            if self.device_map == "auto":
                inputs.input_ids = inputs.input_ids.to("cuda")
            else:
                inputs.input_ids = inputs.input_ids.to(self.model.device)

        temperature = gen_kwargs.get("temperature", 0)
        gen_params = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 1024),
            "do_sample": temperature > 0,
            "use_cache": self.use_cache,
        }
        if temperature > 0:
            gen_params["temperature"] = temperature
            gen_params["top_p"] = gen_kwargs.get("top_p", None)
        if self.eot_token_id is not None:
            gen_params["eos_token_id"] = self.eot_token_id
            gen_params["pad_token_id"] = self.tokenizer.pad_token_id

        generate_kwargs = {
            "input_ids": inputs.input_ids,
            "media": getattr(inputs, "media", None),
            "media_config": getattr(inputs, "media_config", None),
            **gen_params,
        }
        if self.generation_config is not None:
            self.generation_config.update(**gen_params)
            for key in list(gen_params.keys()):
                generate_kwargs.pop(key, None)
            generate_kwargs["generation_config"] = self.generation_config

        outputs = self.model.generate(**generate_kwargs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
