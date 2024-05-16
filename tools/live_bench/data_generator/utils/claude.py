from PIL import Image
import io
import base64
from lmms_eval.live_bench.data_generator.response import Response
import logging
from time import sleep

logger = logging.getLogger("lmms-eval")


def format_claude_images(image: Image.Image):
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


def claude_generate_response(client, model, messages, system, max_tokens: int, max_try_times, **kwargs):
    messages.append({"role": "assistant", "content": "{"})

    def _generate():
        return client.messages.create(model=model, messages=messages, max_tokens=max_tokens, system=system, **kwargs)

    for times in range(max_try_times):
        try:
            response = _generate()
            return Response(success=True, content="{" + response.content[0].text, full_log={"input": messages, "output": response})
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            if times < max_try_times - 1:
                logger.info(f"Retrying... ({times+1}/{max_try_times})")
                sleep(3)
            else:
                logger.error("Failed to generate response after retrying.")
                return Response(success=False, content=str(e), full_log={"input": messages, "output": None})
