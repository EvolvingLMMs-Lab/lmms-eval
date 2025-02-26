import os
os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'
os.environ['USE_SPEECH'] = '1'
import logging
from typing import List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from tqdm import tqdm
from decord import VideoReader, cpu
from datetime import timedelta
from transformers import AutoConfig
import copy
import PIL
import librosa

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from lmms_eval.models.model_utils.audio_processing import downsample_audio

import soundfile as sf
eval_logger = logging.getLogger("lmms-eval")

import sys
wd = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(os.path.join(str(wd), "Ola"))

import whisper
from ola.model.builder import load_pretrained_model
from ola.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
from ola.conversation import conv_templates, SeparatorStyle
from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_token, tokenizer_speech_image_token

try:
    from ola.model.language_model.ola_qwen import OlaConfigQwen
    AutoConfig.register("ola_qwen", OlaConfigQwen)
except:
    eval_logger.debug("")
import moviepy.editor as mp

if "USE_SPEECH" in os.environ:
    USE_SPEECH = os.environ['USE_SPEECH']
    print("USE_SPEECH is set")
else:
    USE_SPEECH = None


@register_model("ola")
class Ola(lmms):
    '''
    How to run lmms-eval with Ola model:

    1. Install Ola: 
    https://github.com/Ola-Omni/Ola?tab=readme-ov-file#installation

    2. Download the pretrained weight from https://huggingface.co/THUdyh/Ola-7b
    or skip this step to use the online weights directly

    3.Download audio encoder from https://huggingface.co/THUdyh/Ola_speech_encoders/tree/main
    and put the weights large-v3.pt and BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt 
    under llms-eval repository (ensure your current directory can see large-v3.pt and BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)

    The path for the project should be like this:
        Folder/contains/lmms-eval/and/Ola  
            ├── lmms-eval (current directory)  
            │   ├── lmms_eval/  
            │   ├── ...  
            │   ├── large-v3.pt  
            │   ├── BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt  
            ├── Ola  
            │   ├── ...  

    4. Run the the command to start evaluate the modeL. For example:
    ```bash
        python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model ola\
        --tasks mme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix mme_ola \
        --output_path ./logs/ 
    ```
    '''
    def __init__(
        self,
        pretrained: str = "THUdyh/Ola-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="",
        conv_template="qwen_1_5",
        use_cache=True,
        truncate_context=False,
        max_frames_num: int = 64,
        mm_resampler_type: str = "spatial_pool",
        overwrite: bool = True,
        video_decode_backend: str = "decord",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

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

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = overwrite
        self.mm_resampler_type = mm_resampler_type
        self.max_frames_num = int(max_frames_num)
        if self.overwrite == True:
            overwrite_config = {}
            overwrite_config["patchify_video_feature"] = False
            overwrite_config["attn_implementation"] = attn_implementation

            cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, device=self.device_map)
        else:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                device_map=self.device_map,
            )

        self._config = self._model.config
        self.model.to('cuda').eval().bfloat16()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
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
        self.accelerator = accelerator
        
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
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())

        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()

        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video = [PIL.Image.fromarray(frame) for frame in spare_frames]
        return video, frame_idx

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

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
                video = self.load_video(visual, self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].bfloat16().to(self.device)
                videos.append(video)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def extract_audio(self, videos_file_path):
        my_clip = mp.VideoFileClip(videos_file_path)
        return my_clip.audio


    def load_audio(self, audio_file_name):
        CHUNK_LIM = 480000
        import librosa
        audio, samplerate = librosa.load(audio_file_name, sr=16000)
        audio = audio.astype(np.float32)

        if len(audio.shape) > 1:
            audio = audio[:, 0]

        speechs = []
        speech_wavs = []
        if len(audio) <= CHUNK_LIM:
            audio = whisper.pad_or_trim(audio)
            speechs.append(audio)
            speech_wavs.append(torch.from_numpy(audio).unsqueeze(0))
        else:
            for i in range(0, len(audio), CHUNK_LIM):
                chunk = audio[i : i + CHUNK_LIM]
                if len(chunk) < CHUNK_LIM:
                    chunk = whisper.pad_or_trim(chunk)
                speechs.append(chunk)
                speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
        mels = []
        for chunk in speechs:
            chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
            mels.append(chunk)

        mels = torch.cat(mels, dim=0)
        speech_wavs = torch.cat(speech_wavs, dim=0)
        if mels.shape[0] > 20:
            mels = mels[:20]
            speech_wavs = speech_wavs[:20]

        speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
        speech_chunks = torch.LongTensor([mels.shape[0]])

        return mels, speech_length, speech_chunks, speech_wavs


    def process_audio(self, audio_array, sampling_rate):
        '''
        Process audio array to format of Ola model
        '''
        audio = audio_array.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        target_sr = 16000
        CHUNK_LIM = 480000
        import librosa
        if sampling_rate != target_sr:
            speech_wav = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=target_sr).astype(np.float32)
        speechs = []
        speech_wavs = []

        if len(speech_wav) <= CHUNK_LIM:
            speech = whisper.pad_or_trim(speech_wav)
            speech_wav = whisper.pad_or_trim(speech_wav)
            speechs.append(speech)
            speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
        else:
            for i in range(0, len(speech_wav), CHUNK_LIM):
                chunk = speech_wav[i : i + CHUNK_LIM]
                if len(chunk) < CHUNK_LIM:
                    chunk = whisper.pad_or_trim(chunk)
                speechs.append(chunk)
                speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
        mels = []
        for chunk in speechs:
            chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
            mels.append(chunk)

        mels = torch.cat(mels, dim=0)
        speech_wavs = torch.cat(speech_wavs, dim=0)
        if mels.shape[0] > 25:
            mels = mels[:25]
            speech_wavs = speech_wavs[:25]

        speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
        speech_chunks = torch.LongTensor([mels.shape[0]])
        return mels, speech_length, speech_chunks, speech_wavs


    def generate_until(self, requests) -> List[str]:
        MODALITY = None
        res = []
        
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            context = contexts[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals) # Len = 1. just an audio tho
            
            speechs, speech_lengths, speech_wavs, speech_chunks = [], [], [], []
            images, images_highres = [], [] # For dummy image passed in audio modality
            image_sizes = []
            image_tensor, image_highres_tensor = [], [] # For image
            video_processed = [] # For video only
            for visual in visuals:
                if isinstance(visual, str): # For Video
                    if MODALITY is None:
                        MODALITY = "VIDEO"
                    # Process audio of video
                    try: 
                        video, frame_idx = self.load_video(visual, self.max_frames_num)
                    except Exception as e:
                        eval_logger.info(f"{e}")
                        eval_logger.info(f"Video {visuals} can not load, check the source")
                        continue
                    audio = self.extract_audio(visual)
                    audio.write_audiofile("./video_audio.wav")
                    video_audio_path = './video_audio.wav'
                    speech, speech_length, speech_chunk, speech_wav = self.load_audio(video_audio_path)
                    speechs.append(speech.bfloat16().to('cuda'))
                    speech_lengths.append(speech_length.to('cuda'))
                    speech_chunks.append(speech_chunk.to('cuda'))
                    speech_wavs.append(speech_wav.to('cuda'))
                    os.remove(video_audio_path)

                    # Process images of video
                    for idx, frame in enumerate(video):
                        self._image_processor.do_resize = False
                        self._image_processor.do_center_crop = False
                        frame = process_anyres_video(frame, self._image_processor)

                        if frame_idx is not None and idx in frame_idx:
                            video_processed.append(frame.unsqueeze(0))
                        elif frame_idx is None:
                            video_processed.append(frame.unsqueeze(0))
                    
                    if frame_idx is None:
                        frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
                    
                    video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
                    video_processed = (video_processed, video_processed)

                    video_data = (video_processed, (384, 384), "video")

                elif isinstance(visual, PIL.Image.Image): # For Image
                    if MODALITY is None:
                        MODALITY = "IMAGE"
                    self._image_processor.do_resize = False
                    self._image_processor.do_center_crop = False
                    image_sizes.append(visual.size)
                    image_tensor_, image_highres_tensor_ = process_anyres_highres_image(visual, self._image_processor)
                    image_tensor.append(image_tensor_)
                    image_highres_tensor.append(image_highres_tensor_)
                    if all(x.shape == image_tensor[0].shape for x in image_tensor):
                        image_tensor = torch.stack(image_tensor, dim=0)
                    if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
                        image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
                    if type(image_tensor) is list:
                        image_tensor = [_image.bfloat16().to("cuda") for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.bfloat16().to("cuda")
                    if type(image_highres_tensor) is list:
                        image_highres_tensor = [_image.bfloat16().to("cuda") for _image in image_highres_tensor]
                    else:
                        image_highres_tensor = image_highres_tensor.bfloat16().to("cuda")

                    # Processing dummy audio, as required by model
                    speechs.append(torch.zeros(1, 3000, 128).bfloat16().to('cuda'))
                    speech_lengths.append(torch.LongTensor([3000]).to('cuda'))
                    speech_wavs.append(torch.zeros([1, 480000]).to('cuda'))
                    speech_chunks.append(torch.LongTensor([1]).to('cuda'))

                elif isinstance(visual, dict) and "array" in visual: # For Audio
                    if MODALITY is None:
                        MODALITY = "AUDIO"
                    mels, speech_length, speech_chunk, speech_wav = self.process_audio(visual['array'], visual['sampling_rate'])
                    speechs.append(mels.bfloat16().to('cuda'))
                    speech_lengths.append(speech_length.to('cuda'))
                    speech_chunks.append(speech_chunk.to('cuda'))
                    speech_wavs.append(speech_wav.to('cuda'))

                    # Processing dummy images, as required by model
                    images.append(torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True))
                    images_highres.append(torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True))
                    image_sizes.append((224, 224))

            if not video_processed and MODALITY == 'VIDEO': 
                # If video is not processed, skip the iteration
                pbar.update(1)
                continue     
                
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            # Okay be I am assuming bs always == 1
            qs = list(contexts)[0]
            if self.model.config.mm_use_im_start_end:
                if MODALITY == "AUDIO":
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_SPEECH_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                elif MODALITY == "IMAGE":
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                elif MODALITY == "VIDEO":
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else: 
                if MODALITY == "AUDIO":
                    qs = DEFAULT_SPEECH_TOKEN + "\n" + qs
                elif MODALITY == "IMAGE":
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                elif MODALITY == "VIDEO":
                    qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + qs


            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{prompt}\n")


            if MODALITY == "AUDIO":
                input_ids = tokenizer_speech_token(prompt, self.tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._device)
            elif MODALITY == "IMAGE":
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._device)
            elif MODALITY == "VIDEO":
                input_ids = tokenizer_speech_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
            pad_token_ids = 151643
            attention_masks = input_ids.ne(pad_token_ids).long().to(self.device)
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1


            try:
                with torch.inference_mode():
                    if MODALITY == "AUDIO":
                        output_ids = self.model.generate(
                            input_ids,
                            images=images,
                            images_highres=images_highres,
                            image_sizes=image_sizes,
                            modalities=['text'],
                            speech=speechs,
                            speech_lengths=speech_lengths,
                            speech_chunks=speech_chunks,
                            speech_wav=speech_wavs,
                            attention_mask=attention_masks,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                            )
                    elif MODALITY == "IMAGE":
                        output_ids = self.model.generate(
                            inputs=input_ids,
                            images=image_tensor,
                            images_highres=image_highres_tensor,
                            image_sizes=image_sizes,
                            modalities=['image'],
                            speech=speechs,
                            speech_lengths=speech_lengths,
                            speech_chunks=speech_chunks,
                            speech_wav=speech_wavs,
                            attention_mask=attention_masks,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                        )
                    elif MODALITY == "VIDEO":
                        output_ids = self.model.generate(
                            inputs=input_ids,
                            images=video_data[0][0],
                            images_highres=video_data[0][1],
                            modalities=video_data[2],
                            speech=speechs,
                            speech_lengths=speech_lengths,
                            speech_chunks=speech_chunks,
                            speech_wav=speech_wavs,
                            attention_mask=attention_masks,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                        )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                outputs = ""
                res.append(outputs)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), outputs)
                continue
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{outputs}\n")

            res.append(outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

