from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages
from lmms_eval.models.model_utils.emu3.emu3_input_processor import Emu3Processor

@register_model("emu3")
class EMU3(lmms):
    """
    EMU3 Chat Model
    https://github.com/baaivision/Emu3
    """

    is_simple = False  # Chat model

    def __init__(
        self,
        pretrained: str = "BAAI/Emu3-Chat",
        vq_hub: str = "BAAI/Emu3-VisionTokenizer",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "flash_attention_2",
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_cache: bool = True,
        max_ratio: int = -1,
        max_pixels: int = 1024 * 1024,
        min_pixels: int = 512 * 512
        **kwargs,
    ):
        super().__init__()
        self.max_ratio = max_ratio # if -1 skip ratio cherck and resize.

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Load main model
        eval_logger.info(f"Loading EMU3 model from {pretrained}")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map=device_map if accelerator.num_processes == 1 else None,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        ).eval()

        # Load tokenizer
        eval_logger.info(f"Loading EMU3 Text Tokenizer from {pretrained}")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code, padding_side="left")

        # Load image processor and image tokenizer
        eval_logger.info(f"Loading EMU3 Vision Tokenizer from {vq_hub}")
        image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=trust_remote_code)
        image_tokenizer = AutoModel.from_pretrained(vq_hub,
                                                    device_map=self.device_map,
                                                    trust_remote_code=trust_remote_code).eval()

        # Create EMU3 processor
        self.processor = Emu3Processor(image_processor, image_tokenizer, self._tokenizer)

        # Set instance variables
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        # Prepare model for distributed training if needed
        if accelerator.num_processes > 1:
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"EMU3 model loaded successfully on rank {self.rank}/{self.world_size}")

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        """Flatten nested lists for handling multiple images."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def preprocess_image_aspect_ratio(self, image: Image.Image, max_ratio: float = 4.9) -> Image.Image:
        """
        Resize image to ensure aspect ratio is within EMU3's constraints.
        EMU3 requires aspect ratio < 5, so we use 4.9 as the safe limit.

        Args:
            image: PIL Image to preprocess
            max_ratio: Maximum allowed aspect ratio (default 4.9 to be safe)

        Returns:
            Preprocessed PIL Image
        """
        width, height = image.size
        aspect_ratio = width / height

        # Check if aspect ratio is within bounds
        if aspect_ratio > max_ratio:
            # Width is too large, resize to fit
            new_width = int(height * max_ratio)
            new_height = height
            eval_logger.warning(f"Resizing image from {width}x{height} (ratio={aspect_ratio:.2f}) to {new_width}x{new_height} (ratio={max_ratio:.2f})")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif aspect_ratio < (1 / max_ratio):
            # Height is too large, resize to fit
            new_width = width
            new_height = int(width * max_ratio)
            eval_logger.warning(f"Resizing image from {width}x{height} (ratio={aspect_ratio:.2f}) to {new_width}x{new_height} (ratio={1/max_ratio:.2f})")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # Group requests by their generation_kwargs
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        # iterate through batches
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            # Convert to ChatMessages protocol
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            # Extract media and text per message
            # EMU3 requires len(images) == len(texts) in understanding mode
            batch_data = []

            for idx, chat_message in enumerate(chat_messages):
                # Extract images for this message
                visual, _, _ = chat_message.extract_media()

                # Extract text for this message
                text = ""
                for message in chat_message.messages:
                    for content in message.content:
                        if content.type == "text":
                            text += content.text

                # EMU3 expects exactly one image per text prompt in understanding mode
                # If there are multiple images, use the first one
                if visual and len(visual) > 0:
                    if len(visual) > 1:
                        eval_logger.warning(f"Sample with multiple images encountered ({len(visual)})! Only take 1st!")
                    img = visual[0]  # Use first image only
                    if isinstance(img, str):
                        img = Image.open(img)
                    # Preprocess image to ensure aspect ratio constraints
                    if self.max_ratio > 0:
                        img = self.preprocess_image_aspect_ratio(img, max_ratio=self.max_ratio)
                    batch_data.append({"text": text, "image": img, "context": ctx[idx]})
                else:
                    # EMU3 requires images in understanding mode, skip text-only samples
                    eval_logger.warning(f"Skipping text-only sample (EMU3 requires images): {text[:50]}...")
                    # Add placeholder answer for this sample
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[0]), "")

            # If all samples were text-only, skip this batch
            if len(batch_data) == 0:
                pbar.update(len(chunk))
                continue

            gen_kwargs = all_gen_kwargs[0]

            # Prepare inputs for EMU3
            texts = [item["text"] for item in batch_data]
            processed_images = [item["image"] for item in batch_data]

            # Process inputs for EMU3
            inputs = self.processor(
                text=texts,
                image=processed_images,
                mode="U",  # Understanding mode
                return_tensors="pt",
                padding="longest",
            )

            # Move to device
            if self.device_map == "auto":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Create generation configuration
            generation_config = GenerationConfig(
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                temperature=gen_kwargs.get("temperature", 0.0),
                do_sample=gen_kwargs.get("do_sample", False),
                top_k=gen_kwargs.get("top_k", None),
                top_p=gen_kwargs.get("top_p", None),
                num_beams=gen_kwargs.get("num_beams", 1),
                use_cache=self.use_cache,
            )

            # Filter inputs to only include keys accepted by model.generate()
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            with torch.no_grad():
                outputs = self.model.generate(**model_inputs, generation_config=generation_config)

            # Trim input_ids from outputs
            outputs_trimmed = outputs[:, model_inputs["input_ids"].shape[-1] :]
            answers = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            for ans, item in zip(answers, batch_data):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (item["context"], gen_kwargs), ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {item['text']}")
                eval_logger.debug(f"Model Response: {ans}")

        # Reorder results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for EMU3")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for EMU3")
