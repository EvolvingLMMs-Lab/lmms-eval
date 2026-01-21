"""
MIO: A Foundation Model on Multimodal Tokens
Paper: https://arxiv.org/abs/2409.17692
HuggingFace: m-a-p/MIO-7B-Instruct

This implementation focuses on image understanding tasks.
For image/speech generation, please refer to the original MIO repository.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple
import torch
from tqdm import tqdm
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils as eval_utils

eval_logger = eval_utils.eval_logger

@register_model("mio")
class MIO(lmms):
    """
    MIO Model for multimodal understanding and generation.
    
    Currently supports:
    - Image Understanding (VQA, Captioning)
    - Video Understanding (frame extraction)
    - Text Generation
    
    TODO: Add support for image/speech generation tasks
    """
    
    def __init__(
        self,
        pretrained: str = "m-a-p/MIO-7B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 1,
        max_new_tokens: int = 512,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.7,
        repetition_penalty: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        self.pretrained = pretrained
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self._batch_size = int(batch_size)
        
        # Generation config
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        eval_logger.info(f"Loading MIO from {pretrained}")
        
        # Check if MIO code is available
        mio_path = Path(__file__).parent.parent.parent.parent / "MIO"
        if not mio_path.exists():
            raise FileNotFoundError(
                f"MIO repository not found at {mio_path}. "
                "Please clone it: git clone https://github.com/MIO-Team/MIO.git"
            )
        
        # Add MIO to path
        sys.path.insert(0, str(mio_path))
        
        try:
            from tokenization_mio import MIOTokenizer
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                f"Failed to import MIO dependencies: {e}\n"
                "Please install requirements: pip install -r MIO/requirements.txt"
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=self._dtype,
            device_map="auto",
        ).eval()
        
        # Load tokenizer
        self.tokenizer = MIOTokenizer(pretrained, str(self._device))
        
        eval_logger.info("✅ MIO model loaded successfully")
    
    @property
    def batch_size(self):
        return self._batch_size
    
    def flatten(self, input_list):
        """Flatten a nested list"""
        return [item for sublist in input_list for item in sublist]
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text responses for image understanding tasks"""
        results = []
        
        for request in tqdm(requests, desc="Processing MIO requests"):
            # Extract context (question + image placeholders)
            context = request.arguments[0]
            
            # Extract images
            visuals = request.arguments[1] if len(request.arguments) > 1 else []
            images = []
            
            if visuals:
                for visual in visuals:
                    if isinstance(visual, str):
                        images.append(Image.open(visual).convert("RGB"))
                    elif isinstance(visual, Image.Image):
                        images.append(visual.convert("RGB"))
            
            # Prepare generation config
            gen_config = {
                "num_beams": self.num_beams,
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
                "eos_token_id": 7,  # <|im_end|> for Instruct model
            }
            
            # Apply chat template (standard mode for image understanding)
            messages = [
                {"role": "system", "content": "You are MIO, an AI assistant capable of understanding and generating images, text, videos, and speech."},
                {"role": "user", "content": context}
            ]
            
            # Tokenize with images
            input_ids = self.tokenizer.tokenize(
                batch_prompts=[context],
                batch_image_paths=[images] if images else None,
                batch_video_paths=None,
                batch_speech_paths=None,
                mode='std',  # Standard mode for understanding
                system_prompt=None,
                apply_chat_template=True
            )
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids.to(self._device),
                    **gen_config
                )
            
            # Decode
            response = self.tokenizer.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            results.append(response.strip())
        
        return results
    
    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Generate responses for multi-round conversations"""
        # TODO: Implement multi-round conversation support
        return self.generate_until(requests)
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for multiple choice tasks"""
        # TODO: Implement if needed for specific benchmarks
        raise NotImplementedError("MIO does not support loglikelihood yet")
