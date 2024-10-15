import logging
from time import sleep

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from live_bench.data_generator.response import Response

logger = logging.getLogger("lmms-eval")


def gemini_generate_response(client: genai.GenerativeModel, messages, max_tokens: int, max_try_times: int = 5, temperature=0.5, **kwargs):
    generation_config = genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)

    def _generate():
        return client.generate_content(
            messages,
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
            **kwargs,
        )

    for times in range(max_try_times):
        try:
            response = _generate()
            return Response(success=True, content=response.text, full_log={"input": messages, "output": response})
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            if times < max_try_times - 1:
                logger.info(f"Retrying... ({times+1}/{max_try_times})")
                sleep(3)
            else:
                logger.error("Failed to generate response after retrying.")
                return Response(success=False, content=str(e), full_log={"input": messages, "output": None})
