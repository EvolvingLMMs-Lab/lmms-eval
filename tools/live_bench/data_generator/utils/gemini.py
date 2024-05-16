import google.generativeai as genai
from time import sleep
from lmms_eval.live_bench.data_generator.response import Response
import logging

logger = logging.getLogger("lmms-eval")


def gemini_generate_response(client: genai.GenerativeModel, messages, max_tokens: int, max_try_times, **kwargs):
    generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)

    def _generate():
        return client.generate_content(messages, generation_config=generation_config, **kwargs)

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
