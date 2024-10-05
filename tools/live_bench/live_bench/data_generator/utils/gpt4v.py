import base64
import io
import logging
import os
from time import sleep

import openai
from live_bench.data_generator.response import Response
from PIL import Image

logger = logging.getLogger("lmms-eval")


def get_openai_client(api_version="2024-02-15-preview") -> openai.OpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return openai.AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return openai.OpenAI(api_key=api_key)


def format_gpt4v_images(image):
    if isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_str}",
            },
        }
    elif isinstance(image, list):
        return [format_gpt4v_images(img) for img in image]
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def format_printable_messages(messages):
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            for content in message["content"]:
                if "type" in content and content["type"] == "image_url":
                    content["image_url"]["url"] = "<image_url>"
    return messages


def gpt4v_generate_response(messages, *, client=None, model="gpt-4o", max_tokens: int = 4096, max_try_times: int = 5, json_format="auto", test=False, system=None, **kwargs) -> Response:
    if system:
        messages = [{"role": "system", "content": system}] + messages

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
        response_format = {"type": "json_object"}
    else:
        response_format = None

    def _generate():
        return client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, response_format=response_format, **kwargs)

    for times in range(max_try_times):
        try:
            response = _generate()
            return Response(success=True, content=response.choices[0].message.content, full_log={"input": format_printable_messages(messages), "output": response.choices[0].message.content})
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            if times < max_try_times - 1:
                logger.info(f"Retrying... ({times+1}/{max_try_times})")
                sleep(3)
            else:
                logger.error("Failed to generate response after retrying.")
                return Response(success=False, content=str(e), full_log={"input": format_printable_messages(messages), "output": None})
