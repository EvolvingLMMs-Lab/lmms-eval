from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests as http_requests
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("openrouter_image_gen")
class OpenRouterImageGen(lmms):
    is_simple = True

    def __init__(
        self,
        model_version: str = "openai/gpt-5-image-mini",
        output_dir: str = "./logs/openrouter_image_gen",
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
        image_size: str = "1024x1024",
        max_retries: int = 3,
        timeout: int = 180,
        **_: Any,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = None if temperature is None else float(temperature)
        self.image_size = image_size
        self.max_retries = max_retries
        self.timeout = timeout

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is required for openrouter_image_gen")

        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.session = http_requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _encode_image(self, image: Image.Image) -> str:
        from io import BytesIO

        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _decode_data_url(self, data_url: str) -> bytes:
        marker = "base64,"
        idx = data_url.find(marker)
        if idx == -1:
            raise ValueError("Image data URL missing base64 payload")
        payload = data_url[idx + len(marker) :]
        return base64.b64decode(payload)

    def _extract_images(self, payload: dict[str, Any]) -> list[str]:
        out: list[str] = []
        try:
            images = payload["choices"][0]["message"].get("images", [])
        except (KeyError, IndexError, TypeError):
            return out

        for item in images:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url", {})
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str) and url.startswith("data:image"):
                out.append(url)
        return out

    def _request_generation(self, prompt: str, visuals: list[Image.Image]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in visuals:
            b64 = self._encode_image(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        payload: dict[str, Any] = {
            "model": self.model_version,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["text", "image"],
            "image": {"size": self.image_size},
            "max_tokens": self.max_new_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(self.base_url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except http_requests.HTTPError as exc:
                detail = ""
                if exc.response is not None:
                    detail = exc.response.text
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenRouter HTTPError: {detail}") from exc
                time.sleep(min(2 * attempt, 8))
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2 * attempt, 8))
        raise RuntimeError("Unreachable retry loop")

    def _save_images(self, image_data_urls: list[str], task: str, doc_id: int) -> list[str]:
        task_dir = Path(self.output_dir) / str(task).replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        for idx, data_url in enumerate(image_data_urls):
            raw = self._decode_data_url(data_url)
            path = task_dir / f"{doc_id}_{idx}.png"
            path.write_bytes(raw)
            saved_paths.append(str(path))
        return saved_paths

    def generate_until(self, requests: list[Instance]) -> list[str]:
        outputs: list[str] = []
        for req in requests:
            args = req.args
            if len(args) < 6:
                outputs.append(json.dumps({"text": "", "images": []}, ensure_ascii=False))
                continue
            ctx, gen_kwargs, doc_to_visual, doc_id, task, split = args[:6]
            prompt = str(ctx)
            local_gen_kwargs = dict(gen_kwargs or {})

            visuals_raw = doc_to_visual(self.task_dict[task][split][doc_id])
            visuals: list[Image.Image] = []
            for item in visuals_raw:
                if isinstance(item, Image.Image):
                    visuals.append(item)

            if "max_new_tokens" in local_gen_kwargs:
                self.max_new_tokens = int(local_gen_kwargs["max_new_tokens"])
            if "temperature" in local_gen_kwargs:
                value = local_gen_kwargs["temperature"]
                self.temperature = None if value is None else float(value)

            try:
                data = self._request_generation(prompt=prompt, visuals=visuals)
            except Exception:
                data = self._request_generation(prompt=prompt, visuals=[])
            image_urls = self._extract_images(data)
            saved_images = self._save_images(image_urls, task=str(task), doc_id=int(doc_id))

            text = ""
            try:
                text = data["choices"][0]["message"].get("content", "")
            except (KeyError, IndexError, TypeError):
                text = ""

            result = {"text": text, "images": saved_images}
            outputs.append(json.dumps(result, ensure_ascii=False))
            self.cache_hook.add_partial("generate_until", (ctx, local_gen_kwargs), outputs[-1])

        return outputs

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("openrouter_image_gen does not support loglikelihood")

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("openrouter_image_gen does not support multi-round generation")
