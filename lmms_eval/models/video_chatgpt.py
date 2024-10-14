import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from huggingface_hub import snapshot_download
from loguru import logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logger

try:
    from lmms_eval.models.video_chatgpt.eval.model_utils import (
        initialize_model,
        load_video,
    )
    from lmms_eval.models.video_chatgpt.inference import (
        get_spatio_temporal_features_torch,
        video_chatgpt_infer,
        video_chatgpt_infer_ppl,
    )
except ImportError:
    eval_logger.warning("Failed to import video_chatgpt modules")

from lmms_eval.models.model_utils.load_video import read_video_pyav


@register_model("video_chatgpt")
class VideoChatGPT(lmms):
    def __init__(
        self,
        batch_size: Optional[Union[int, str]] = 1,
        projection_path: str = "MBZUAI/Video-ChatGPT-7B",
        model_path: str = "mmaaz60/LLaVA-7B-Lightening-v1-1",
        device_map="cuda:0",
        device: Optional[str] = "cuda:0",
        num_frm: Optional[Union[int, str]] = 100,
    ) -> None:
        super().__init__()
        self.batch_size_per_gpu = int(batch_size)
        self.num_frm = int(num_frm)
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        try:
            self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(model_path, projection_path, device=self.device)
        except:
            eval_logger.info("Does not find the model from the path you provide, try downloading from the hf repo.")
            model_path = snapshot_download(repo_id=model_path)
            projection_path = os.path.join(snapshot_download(repo_id=projection_path), "video_chatgpt-7B.bin")
            self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(model_path, projection_path, device=self.device)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            # videos = []
            for visual in visuals:
                video_frames = read_video_pyav(visual, num_frm=self.num_frm)
                target_h, target_w = 224, 224
                # If image shape is not as target, resize it
                if video_frames.shape[-3] != target_h or video_frames.shape[-2] != target_w:
                    video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()
                    video_frames = torch.nn.functional.interpolate(video_frames, size=(target_h, target_w))
                    video_frames = video_frames.permute(0, 2, 3, 1).to(torch.uint8).numpy()
                video_frames = [Image.fromarray(frame) for frame in video_frames]
                if len(video_frames) > self.num_frm:
                    video_frames = video_frames[: self.num_frm]
                # VideoChatGPT load video return a list of PIL Image
                # videos += video_frames

            output = video_chatgpt_infer(
                video_frames, contexts, conv_mode="video-chatgpt_v1", model=self.model, vision_tower=self.vision_tower, tokenizer=self.tokenizer, image_processor=self.image_processor, video_token_len=self.video_token_len
            )

            res.append(output)
            pbar.update(1)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video_frames = load_video(visual, num_frm=self.num_frm)
                # VideoChatGPT load video return a list of PIL Image
                videos += video_frames
            image_tensor = self.image_processor.preprocess(videos, return_tensors="pt")["pixel_values"]

            # Move image tensor to GPU and reduce precision to half
            image_tensor = image_tensor.half().to(self.device)

            # Generate video spatio-temporal features
            with torch.no_grad():
                image_forward_outs = self.vision_tower(image_tensor, output_hidden_states=True)
                frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # Use second to last layer as in LLaVA
            video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features).cuda()

            outputs, input_ids, context_ids = video_chatgpt_infer_ppl(
                # video_frames,
                contexts,
                continuation,
                conv_mode="video-chatgpt_v1",
                model=self.model,
                vision_tower=self.vision_tower,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                video_token_len=self.video_token_len,
                video_spatio_temporal_features=video_spatio_temporal_features,
            )

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, context_ids.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, context_ids.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

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

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
