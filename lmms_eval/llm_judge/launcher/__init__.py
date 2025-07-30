import sys

from loguru import logger

logger.remove()
# Configure logger with detailed format including file path, function name, and line number
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
logger.add(sys.stdout, level="WARNING", format=log_format)

AVAILABLE_LAUNCHERS = {"sglang": "SGLangLauncher"}


def get_launcher(name):
    """
    Get a launcher class by its name.

    :param name: The name of the launcher.
    :return: The launcher class if found, otherwise None.
    """
    model_class = AVAILABLE_LAUNCHERS[name]
    if "." not in model_class:
        model_class = f"lmms_eval.llm_judge.launcher.{name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {name}: {e}")
        raise


__all__ = [
    "get_launcher",
    "AVAILABLE_LAUNCHERS",
]
