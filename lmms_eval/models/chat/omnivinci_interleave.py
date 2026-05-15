"""omnivinci_interleave — OmniVinci over interleaved doc_to_messages.

See `_interleave_base.InterleaveChatMixin` for the shared rationale and
request loop. Only the OmniVinci/VILA-specific inference step lives here.
"""

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._interleave_base import InterleaveChatMixin
from lmms_eval.models.simple.omnivinci import OmniVinci


@register_model("omnivinci_interleave")
class OmniVinciInterleave(InterleaveChatMixin, OmniVinci):
    """OmniVinci that consumes interleaved doc_to_messages prompts."""

    def _build_message(self, messages: list) -> list:
        """doc_to_messages blocks -> OmniVinci message (string system + user list)."""
        user_content = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                t = c.get("type")
                if t == "text":
                    user_content.append({"type": "text", "text": c.get("text", "")})
                elif t == "image":
                    user_content.append({"type": "image", "image": c["url"]})
                elif t == "audio":
                    user_content.append({"type": "audio", "audio": c["url"]})
                elif t == "video":
                    user_content.append({"type": "video", "video": c["url"]})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _infer_one(self, messages: list, gen_kwargs: dict) -> str:
        message = self._build_message(messages)

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
