import base64
import io
import logging
from time import sleep
from typing import List, Union

import anthropic
from live_bench.data_generator.response import Response
from PIL import Image

logger = logging.getLogger("lmms-eval")


def format_claude_images(image: Union[Image.Image, List[Image.Image]]):
    if isinstance(image, list):
        return [format_claude_images(img) for img in image]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": img_str,
        },
    }


def claude_generate_response(client: anthropic.Anthropic, model, messages, max_tokens: int = 4096, max_try_times=5, system=None, json_format="auto", test=False, tempreture=0.5, **kwargs):
    if json_format == "auto":
        json_format = False
        for message in messages:
            if message.get("role") == "user":
                contents = message.get("content", [])
                if isinstance(contents, str):
                    if "json" in contents:
                        json_format = True
                        break
                else:
                    for content in contents:
                        if content.get("type", None) == "text" and "json" in content.get("text", ""):
                            json_format = True
                            break

    if json_format:
        messages.append({"role": "assistant", "content": "{"})

    def _generate():
        if system:
            return client.messages.create(model=model, messages=messages, max_tokens=max_tokens, system=system, temperature=tempreture, **kwargs)
        else:
            return client.messages.create(model=model, messages=messages, max_tokens=max_tokens, temperature=tempreture, **kwargs)

    for times in range(max_try_times):
        try:
            response = _generate()
            response_str = response.content[0].text
            if json_format:
                response_str = "{" + response_str
            return Response(success=True, content=response_str, full_log={"input": messages, "output": response.to_dict()})
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            if times < max_try_times - 1:
                logger.info(f"Retrying... ({times+1}/{max_try_times})")
                sleep(3)
            else:
                logger.error("Failed to generate response after retrying.")
                return Response(success=False, content=str(e), full_log={"input": messages, "output": None})
