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
            # Extract arguments from request
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            
            # Get document and extract images
            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)] if doc_to_visual else []
            visuals = self.flatten(visuals)
            
            # Prepare image paths - keep temp files alive
            image_paths = []
            temp_files = []  # Keep reference to prevent deletion
            if visuals:
                for i, visual in enumerate(visuals):
                    if isinstance(visual, str):
                        image_paths.append(visual)
                    else:
                        # Handle PIL Image objects - save to persistent temp file
                        import tempfile
                        from PIL import Image
                        if isinstance(visual, Image.Image):
                            # Create temp file with delete=False to keep it
                            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                            visual.save(tmp.name, 'JPEG')
                            tmp.close()
                            image_paths.append(tmp.name)
                            temp_files.append(tmp.name)
                            if doc_id < 3:  # Debug first 3 samples
                                eval_logger.info(f"[Doc {doc_id}] Created temp image {i}: {tmp.name} (size: {visual.size})")
                                import os
                                eval_logger.info(f"[Doc {doc_id}] Temp file exists: {os.path.exists(tmp.name)}")
            
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
            
            # Add image placeholders to context if images exist
            if image_paths:
                # Add image placeholders at the end of the question
                image_placeholders = "".join([f"<image_placeholder_{i}>" for i in range(len(image_paths))])
                context_with_images = f"{context}\n{image_placeholders}"
                eval_logger.info(f"[Doc {doc_id}] 🖼️ Images: {len(image_paths)} paths: {[p[:50] for p in image_paths]}")
                eval_logger.info(f"[Doc {doc_id}] 📝 Context: {context_with_images[:150]}")
            else:
                context_with_images = context
                eval_logger.info(f"[Doc {doc_id}] ❌ NO IMAGES for context: {context[:100]}")
            
            # Prepare conversation in MIO format
            conversations = [[{"role": "user", "content": context_with_images}]]
            
            # Apply chat template (standard mode for image understanding)
            # Note: batch_image_paths expects a list of image path lists for each conversation
            batch_img_paths = [image_paths] if image_paths else None
            eval_logger.info(f"[Doc {doc_id}] 🔄 Calling tokenizer with batch_image_paths: {batch_img_paths}")
            
            inputs = self.tokenizer.apply_chat_template(
                conversations,
                batch_image_paths=batch_img_paths,
                batch_speech_paths=None,
                mode='std',
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self._device)
            attention_mask = inputs['attention_mask'].to(self._device)
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config
                )
            
            # Decode using MIO's detokenize method
            generated_sequences, _, _ = self.tokenizer.detokenize(
                outputs,
                output_image_dir=None,
                output_speech_dir=None,
                extract_assistant=True,  # Extract only assistant response for Instruct model
                save_images=False,
                save_speeches=False
            )
            
            response = generated_sequences[0].strip()
            
            # Clean up any image/speech tokens that model generated incorrectly
            import re
            # Remove image tokens like <img749>, <image>, </image>
            response = re.sub(r'<img\d+>', '', response)
            response = re.sub(r'</?image>', '', response)
            # Remove speech tokens like <spch749>, <spch>, </spch>
            response = re.sub(r'<spch\d+>', '', response)
            response = re.sub(r'</?spch>', '', response)
            response = response.strip()
            
            if doc_id < 5:
                eval_logger.info(f"[Doc {doc_id}] 💬 Response: '{response[:100]}' | Target: '{doc.get('target', 'N/A')}'")
            results.append(response)
            
            # Clean up temp files
            import os
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        return results
    
    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Generate responses for multi-round conversations"""
        # TODO: Implement multi-round conversation support
        return self.generate_until(requests)
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for multiple choice tasks"""
        # TODO: Implement if needed for specific benchmarks
        raise NotImplementedError("MIO does not support loglikelihood yet")
