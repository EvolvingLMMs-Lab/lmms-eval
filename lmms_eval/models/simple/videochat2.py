import logging
import os
from typing import List, Tuple

import decord
import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from easydict import EasyDict

decord.bridge.set_bridge("torch")
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")


from datetime import timedelta

from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


def get_index(max_frame, num_segments, fps, first_idx=0, bound=None):
    if bound:
        start, end = bound[0], bound[1]
        if start is None:
            start, end = -100000, 100000
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_image(image_path, resolution=224, hd_num=6):
    image = Image.open(image_path).convert("RGB")
    image_tensor = T.PILToTensor()(image).unsqueeze(0)
    image_tensor = HD_transform_no_padding(image_tensor.float(), image_size=resolution, hd_num=hd_num)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = T.Compose([T.Lambda(lambda x: x.float().div(255.0)), T.Normalize(mean, std)])
    image_tensor = transform(image_tensor).unsqueeze(0)

    return image_tensor


def load_video(video_path, num_segments=16, return_msg=False, resolution=224, hd_num=6, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr) - 1

    frame_indices = get_index(max_frame=num_frames, num_segments=num_segments, fps=float(vr.get_avg_fps()), first_idx=0, bound=None)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = T.Compose([T.Lambda(lambda x: x.float().div(255.0)), T.Normalize(mean, std)])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames).unsqueeze(0)

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames


def HD_transform_padding(frames, image_size=224, hd_num=6):
    def _padding_224(frames):
        _, _, H, W = frames.shape
        tar = int(np.ceil(H / 224) * 224)
        top_padding = (tar - H) // 2
        bottom_padding = tar - H - top_padding
        left_padding = 0
        right_padding = 0

        padded_frames = F.pad(frames, pad=[left_padding, right_padding, top_padding, bottom_padding], mode="constant", value=255)
        return padded_frames

    _, _, H, W = frames.shape
    trans = False
    if W < H:
        frames = frames.flip(-2, -1)
        trans = True
        width, height = H, W
    else:
        width, height = W, H

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)

    resized_frames = F.interpolate(frames, size=(new_h, new_w), mode="bicubic", align_corners=False)
    padded_frames = _padding_224(resized_frames)

    if trans:
        padded_frames = padded_frames.flip(-2, -1)

    return padded_frames


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2, 1)):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    # calculate the existing video aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the frames
    resized_frame = F.interpolate(frames, size=(target_height, target_width), mode="bicubic", align_corners=False)
    return resized_frame


@register_model("VideoChat2")
class VideoChat2(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_segments: str = "16",
        hd_num: str = "6",
        **kwargs,
    ):
        super().__init__()
        self.path = pretrained
        self.instruction = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons.\n"

        # self._tokenizer =  AutoTokenizer.from_pretrained(self.path,
        #                 trust_remote_code=True,
        #                 use_fast=False)
        self._model = (
            AutoModel.from_pretrained(
                self.path,
                # torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        self._tokenizer = self._model.mistral_tokenizer
        batch_size = int(batch_size)
        self.num_segments = int(num_segments)
        self.hd_num = int(hd_num)
        assert batch_size == 1, f"Batch size should be 1 for InternVideo2, but got {batch_size}."
        self.batch_size_per_gpu = batch_size
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

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
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

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

    def get_prompt(self, conv):
        ret = conv.system + conv.sep
        for role, message in conv.messages:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
        return ret

    def get_prompt2(self, conv):
        ret = conv.system + conv.sep
        count = 0
        for role, message in conv.messages:
            count += 1
            if count == len(conv.messages):
                ret += role + " " + message
            else:
                if message:
                    ret += role + " " + message + " " + conv.sep
                else:
                    ret += role
        return ret

    def get_context_emb(self, conv, img_list, answer_prompt=None, print_res=False):
        if answer_prompt:
            prompt = self.get_prompt2(conv)
        else:
            prompt = self.get_prompt(conv)
        if print_res:
            print(prompt)
        if "<VideoHere>" in prompt:
            prompt_segs = prompt.split("<VideoHere>")
        else:
            prompt_segs = prompt.split("<ImageHere>")
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        with torch.no_grad():
            seg_tokens = [
                self.model.mistral_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).to(self._device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [self.model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def get_sinusoid_encoding_table(self, n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784):
        """Sinusoid position encoding table"""

        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # generate checkpoint position embedding
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

        # print(f"n_position: {n_position}")
        # print(f"pre_n_position: {pre_n_position}")

        if n_position != pre_n_position:
            T = ckpt_num_frame  # checkpoint frame
            P = 14  # checkpoint size
            C = d_hid
            new_P = int((n_position // cur_frame) ** 0.5)  # testing size
            if new_P != 14:
                # print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
                # print(f'Interpolate the position embedding')
                sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
                sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
                sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=(new_P, new_P), mode="bicubic", align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
                sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

        if cur_frame != ckpt_num_frame:
            # print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
            # print(f'Interpolate the position embedding')
            T = ckpt_num_frame  # checkpoint frame
            new_T = cur_frame  # testing frame
            # interpolate
            P = int((n_position // cur_frame) ** 0.5)  # testing size
            C = d_hid
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
            sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode="linear")
            sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3)  # B, T, H, W, C
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

        return sinusoid_table

    def ask(self, text, conv):
        conv.messages.append([conv.roles[0], text])

    def answer(self, conv, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
        stop_words_ids = [torch.tensor([2]).to(self._device), torch.tensor([29871, 2]).to(self._device)]  # '</s>' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        conv.messages.append([conv.roles[1], answer_prompt])
        embs = self.get_context_emb(conv, img_list, answer_prompt=answer_prompt, print_res=print_res)

        with torch.no_grad():
            outputs = self.model.mistral_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split("</s>")[0]  # remove the stop sign </s>
        conv.messages[-1][1] = output_text + "</s>"
        return output_text, output_token.cpu().numpy()

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
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if self.modality == "image":
                image_path = visuals[0]
                new_pos_emb = self.get_sinusoid_encoding_table(n_position=(224 // 16) ** 2, cur_frame=1, ckpt_num_frame=1, pre_n_position=14 * 14)
                self.model.vision_encoder.encoder.img_pos_embed = new_pos_emb
                pixel_values = load_image(image_path, resolution=224, hd_num=self.hd_num)
                pixel_values = pixel_values.cuda()
                question = self.instruction + contexts
                with torch.no_grad():
                    img_list = []
                    image_emb, _, _ = self.model.encode_img(pixel_values, self.instruction + "Observe the image and answer the question.")
                    img_list.append(image_emb[0])
                    chat = EasyDict({"system": "", "roles": ("[INST]", "[/INST]"), "messages": [], "sep": ""})

                    chat.messages.append([chat.roles[0], "<Image><ImageHere></Image> [/INST]"])
                    self.ask(question, chat)
                    response = self.answer(conv=chat, do_sample=False, img_list=img_list, max_new_tokens=512, answer_prompt=answer_prompt)[0]
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos. [META-INFO]{visuals}"
                video_path = visuals[0]
                if "mvbench" in task:
                    answer_prompt = "Best Option:("
                else:
                    answer_prompt = None
                new_pos_emb = self.get_sinusoid_encoding_table(n_position=(224 // 16) ** 2 * self.num_segments, cur_frame=self.num_segments)
                self.model.vision_encoder.encoder.pos_embed = new_pos_emb
                pixel_values = load_video(video_path, num_segments=self.num_segments, return_msg=False, resolution=224, hd_num=self.hd_num)
                pixel_values = pixel_values.cuda()
                question = self.instruction + contexts
                with torch.no_grad():
                    img_list = []
                    image_emb, _, _ = self.model.encode_img(pixel_values, self.instruction + "Watch the video and answer the question.")
                    img_list.append(image_emb[0])
                    chat = EasyDict({"system": "", "roles": ("[INST]", "[/INST]"), "messages": [], "sep": ""})

                    chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
                    self.ask(question, chat)
                    response = self.answer(conv=chat, do_sample=False, img_list=img_list, max_new_tokens=512, answer_prompt=answer_prompt)[0]
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for VideoChat2")
