import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


def _encode_image_to_base64(image: Image.Image) -> str:
    img = image.convert("RGB")
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_data_url(data_url: str) -> Optional[bytes]:
    if not data_url.startswith("data:image"):
        return None
    try:
        b64 = data_url.split(",", 1)[1]
        return base64.b64decode(b64)
    except Exception:
        return None


@register_model("openrouter_image_gen")
class OpenRouterImageGen(lmms):
    is_simple = True

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash-image",
        output_dir: str = "./logs/openrouter_image_gen",
        num_concurrent: int = 2,
        max_tokens: int = 800,
        image_size: str = "1024x1024",
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.num_concurrent = max(1, int(num_concurrent))
        self.max_tokens = int(max_tokens)
        self.image_size = image_size
        self.temperature = float(temperature)
        self.base_url = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY or OPENROUTER_API_KEY is required for openrouter_image_gen")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        eval_logger.info(f"OpenRouterImageGen initialized: model={self.model}, output_dir={self.output_dir}")

    def _request_image(self, prompt: str, input_images: List[Image.Image]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in input_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_encode_image_to_base64(img)}"},
                }
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["text", "image"],
            "image": {"size": self.image_size},
            "max_tokens": self.max_tokens,
        }
        if "gpt-5" not in self.model:
            payload["temperature"] = self.temperature

        resp = self._session.post(f"{self.base_url}/chat/completions", json=payload, timeout=300)
        if resp.status_code >= 400:
            raise RuntimeError(f"openrouter error {resp.status_code}: {resp.text[:2000]}")
        return resp.json()

    def _extract_images(self, response: Dict[str, Any]) -> List[bytes]:
        out: List[bytes] = []
        msg = response.get("choices", [{}])[0].get("message", {})
        for item in msg.get("images", []) or []:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            data = _decode_data_url(url) if isinstance(url, str) else None
            if data is not None:
                out.append(data)
        return out

    def _generate_one(self, req: Instance) -> str:
        args = req.args
        if len(args) < 6:
            return json.dumps({"text": "", "images": [], "error": "invalid request args"})

        ctx = args[0]
        doc_to_visual = args[2]
        doc_id = args[3]
        task = args[4]
        split = args[5]
        prompt = str(ctx).strip()
        if not prompt:
            return json.dumps({"text": "", "images": [], "error": "empty prompt"})

        doc = self.task_dict[task][split][doc_id]
        visuals_raw = doc_to_visual(doc) if doc_to_visual else []
        visuals: List[Image.Image] = []
        for v in visuals_raw or []:
            if isinstance(v, Image.Image):
                visuals.append(v.convert("RGB"))

        response = self._request_image(prompt, visuals)
        image_blobs = self._extract_images(response)
        task_dir = Path(self.output_dir) / str(task)
        task_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[str] = []
        for idx, blob in enumerate(image_blobs):
            out_path = task_dir / f"{task}_{doc_id}_{idx}.png"
            out_path.write_bytes(blob)
            saved_paths.append(str(out_path))

        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return json.dumps({"text": text, "images": saved_paths}, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Generating Images")

        with ThreadPoolExecutor(max_workers=self.num_concurrent) as pool:
            futures = {pool.submit(self._generate_one, req): i for i, req in enumerate(requests)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    eval_logger.error(f"Generation failed for index={idx}: {exc}")
                    results[idx] = json.dumps({"text": "", "images": [], "error": str(exc)})
                pbar.update(1)
        pbar.close()
        return [r if r is not None else json.dumps({"text": "", "images": [], "error": "unknown"}) for r in results]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("OpenRouterImageGen does not support loglikelihood")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("OpenRouterImageGen does not support multi-round generation")
