"""
Utility for loading Emu3.5 tokenizer.

The BAAI/Emu3.5 HuggingFace repo references tokenization_emu3.Emu3Tokenizer
in tokenizer_config.json, but the tokenization_emu3.py file may be missing
from the repo. This utility provides fallback loading.
"""

import os
import os.path as osp
from pathlib import Path

from loguru import logger as eval_logger
from transformers import AutoTokenizer


def load_emu3_tokenizer(
    pretrained_path: str, trust_remote_code: bool = True, **kwargs
):
    """
    Load Emu3.5 tokenizer matching the official implementation.

    This function replicates the tokenizer loading from
    external/Emu3.5/src/utils/model_utils.py, including:
    - Loading with special_tokens_file parameter
    - Setting all required special tokens manually

    Args:
        pretrained_path: Path to the pretrained model (HF model ID or local path)
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for AutoTokenizer.from_pretrained

    Returns:
        Loaded tokenizer with all special tokens configured

    Raises:
        ValueError: If tokenizer cannot be loaded and no fallback is available
    """
    try:
        eval_logger.info(
            f"Loading Emu3.5 tokenizer from {pretrained_path} with "
            f"trust_remote_code={trust_remote_code}"
        )

        # Check if special_tokens_file exists at pretrained_path
        special_tokens_path = None
        if os.path.exists(pretrained_path):
            # Local path - check for emu3_vision_tokens.txt
            potential_path = osp.join(pretrained_path, "emu3_vision_tokens.txt")
            if os.path.exists(potential_path):
                special_tokens_path = potential_path
                eval_logger.info(f"Found special tokens file at {special_tokens_path}")

        # Load tokenizer (with special_tokens_file if available)
        if special_tokens_path:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path,
                special_tokens_file=special_tokens_path,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        else:
            # HF path or missing file - load without special_tokens_file
            # The file should be downloaded automatically from HF
            eval_logger.info(
                "Loading from HuggingFace - special_tokens_file should be "
                "downloaded automatically"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path, trust_remote_code=trust_remote_code, **kwargs
            )

        # Set special tokens as done in official implementation
        # See external/Emu3.5/src/utils/model_utils.py lines 49-63
        tokenizer.bos_token = "<|extra_203|>"
        tokenizer.eos_token = "<|extra_204|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eol_token = "<|extra_200|>"
        tokenizer.eof_token = "<|extra_201|>"
        tokenizer.tms_token = "<|extra_202|>"
        tokenizer.img_token = "<|image token|>"
        tokenizer.boi_token = "<|image start|>"
        tokenizer.eoi_token = "<|image end|>"
        tokenizer.bss_token = "<|extra_100|>"
        tokenizer.ess_token = "<|extra_101|>"
        tokenizer.bog_token = "<|extra_60|>"
        tokenizer.eog_token = "<|extra_61|>"
        tokenizer.boc_token = "<|extra_50|>"
        tokenizer.eoc_token = "<|extra_51|>"

        eval_logger.info("Emu3.5 tokenizer loaded successfully with special tokens configured")
        return tokenizer

    except Exception as e:
        eval_logger.warning(
            f"Failed to load tokenizer from {pretrained_path}: {e}"
        )

        # Check if we're loading from HF and the issue is missing tokenization file
        if "tokenization_emu3" in str(e) or "Emu3Tokenizer" in str(e):
            raise ValueError(
                f"The Emu3.5 tokenizer could not be loaded from {pretrained_path}. "
                "The HuggingFace repository appears to be missing the tokenization_emu3.py file. "
                "\n\nTo fix this issue:\n"
                "1. Clone the Emu3.5 repository: "
                "git clone https://github.com/baaivision/Emu3.5\n"
                "2. Copy the tokenization file to your model directory:\n"
                f"   cp Emu3.5/src/tokenizer_emu3_ibq/tokenization_emu3.py "
                "<your-model-path>/\n"
                "3. Also copy the required tokenizer files:\n"
                "   - emu3.tiktoken\n"
                "   - emu3_vision_tokens.txt\n"
                "   - tokenizer_config.json\n"
                "4. Load from the local path instead of HF model ID\n\n"
                "Alternatively, wait for the HuggingFace repository to be updated "
                "with the missing file."
            ) from e
        else:
            raise ValueError(
                f"Failed to load Emu3.5 tokenizer: {e}"
            ) from e


__all__ = ["load_emu3_tokenizer"]