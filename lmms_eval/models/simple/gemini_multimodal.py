import io
import json
import os
import time
from typing import List, Tuple, Dict, Any
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from google import genai
    from google.genai import types
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        eval_logger.warning("GOOGLE_API_KEY is not set")
except Exception as e:
    eval_logger.error(f"Failed to import google-genai: {str(e)}")
    eval_logger.error("Please install: pip install google-genai")
    genai = None
    types = None


@register_model("gemini_multimodal")
class GeminiMultimodal(lmms):
    """
    Gemini 
    
    - input: text
    - output: text + image
    
    example:
    accelerate launch -m lmms_eval \
        --model gemini_multimodal \
        --model_args model_version=gemini-2.5-flash-image-preview,enable_image_generation=True \
        --tasks your_task \
        --batch_size 1 \
        --output_path ./logs/
    """
    
    def __init__(
        self,
        model_version: str = "gemini-2.5-flash-image-preview",
        enable_image_generation: bool = True,
        output_image_dir: str = None,
        response_modalities: List[str] = None,
        continual_mode: bool = True,
        response_persistent_folder: str = None,
        max_retries: int = 5,
        retry_delay: int = 3,
        structured_output: bool = True,  # 是否返回结构化JSON字符串（推荐）
        **kwargs,
    ) -> None:
        super().__init__()
        
        if genai is None:
            raise ImportError(
                "需要安装 google-genai。请运行: pip install google-genai"
            )
        
        self.model_version = model_version
        self.enable_image_generation = enable_image_generation
        self.continual_mode = continual_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.structured_output = structured_output
        
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        
        if response_modalities is None:
            if enable_image_generation:
                self.response_modalities = ['TEXT', 'IMAGE']
            else:
                self.response_modalities = ['TEXT']
        else:
            self.response_modalities = response_modalities
        

        self.response_cache = {}
        self.cache_mode = "start"
        

        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/gemini_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(self.response_persistent_folder, "gemini_generated_images")
        else:
            self.output_image_dir = output_image_dir
        
        if self.enable_image_generation:
            os.makedirs(self.output_image_dir, exist_ok=True)
            eval_logger.info(f"Image output directory: {self.output_image_dir}")
        
        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, 
                f"{self.model_version.replace('/', '_')}_response.json"
            )
            
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Load cache: {len(self.response_cache)} records")
        
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continuous mode is not supported for distributed inference"
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1
        
        eval_logger.info(f"Initialize Gemini multimodal model: {model_version}")
        eval_logger.info(f"Response modalities: {self.response_modalities}")
        eval_logger.info(f"Structured output: {self.structured_output}")
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    def flatten(self, input_list):
        new_list = []
        for i in input_list:
            if isinstance(i, list):
                for j in i:
                    new_list.append(j)
            else:
                new_list.append(i)
        return new_list
    
    def generate_with_retry(
        self, 
        prompt: str, 
        doc_id: str,
        task: str
    ) -> Tuple[str, List[str], List[Dict]]:
        """
        Generate with retry
        
        Returns:
            tuple: (text, image_paths, interleaved_parts)
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                eval_logger.debug(f"Attempt {attempt} to generate content...")
                
                response = self.client.models.generate_content(
                    model=self.model_version,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=self.response_modalities
                    )
                )

                if not response or not response.candidates:
                    eval_logger.warning(f"Attempt {attempt} failed: response is invalid")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                        continue
                    return "", [], []
                
                if not response.candidates[0].content.parts:
                    eval_logger.warning(f"Attempt {attempt} failed: response content is empty")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                        continue
                    return "", [], []
                
                output_text, output_images, interleaved_parts = self.process_response(
                    response, doc_id, task
                )
                
                if self.enable_image_generation:
                    if output_text and output_images:
                        eval_logger.info(
                            f"Attempt {attempt} succeeded! "
                            f"Text: {len(output_text)} characters, "
                            f"Images: {len(output_images)}"
                        )
                        return output_text, output_images, interleaved_parts
                    else:
                        eval_logger.warning(f"Attempt {attempt} failed: no valid content generated")
                else:
                    if output_text:
                        return output_text, output_images, interleaved_parts
                    else:
                        eval_logger.warning(f"Attempt {attempt} failed: no text generated")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                eval_logger.error(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    eval_logger.error(f"All {self.max_retries} attempts failed")
        
        return "", [], []
    
    def process_response(
        self, 
        response, 
        doc_id: str, 
        task: str
    ) -> Tuple[str, List[str], List[Dict]]:
        """
        Process Gemini response, extract text and images
        
        Returns:
            tuple: (output_text, output_images, interleaved_parts)
        """
        output_text = ""
        output_images = []
        interleaved_parts = []
        image_counter = 1
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text_content = part.text
                output_text += text_content
                interleaved_parts.append({
                    "type": "text",
                    "content": text_content
                })
                eval_logger.debug(f"Generated text: {text_content[:100]}...")
            
            if part.inline_data is not None:
                try:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    
                    safe_filename = f"{task}_{doc_id}_{image_counter}.png"
                    image_path = os.path.join(self.output_image_dir, safe_filename)
                    
                    image.save(image_path)
                    output_images.append(image_path)
                    
                    interleaved_parts.append({
                        "type": "image",
                        "path": image_path,
                        "index": image_counter - 1
                    })
                    
                    eval_logger.info(f"Saved image {image_counter}: {image_path}")
                    image_counter += 1
                    
                except Exception as e:
                    eval_logger.error(f"Failed to save image: {e}")
        
        return output_text, output_images, interleaved_parts
    
    def format_output(
        self, 
        text: str, 
        images: List[str]
    ) -> str:
   
        output_dict = {
            "text": text,
            "images": images
        }
        return json.dumps(output_dict, ensure_ascii=False)
    
    def generate_until(self, requests) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests), 
            disable=(self.rank != 0), 
            desc="Gemini Multimodal Generating"
        )
        
        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"
        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            doc_uuid = get_uuid(task, split, doc_id)
            
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue
            

            prompt = contexts
            
            output_text, output_images, interleaved_parts = self.generate_with_retry(
                prompt, str(doc_id), task
            )
            
            formatted_output = self.format_output(output_text, output_images)
            
            res.append(formatted_output)
            
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
            
            pbar.update(1)
        
        pbar.close()
        return res
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported loglikelihood"""
        raise NotImplementedError("Gemini multimodal generation model does not support loglikelihood")
    
    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")
