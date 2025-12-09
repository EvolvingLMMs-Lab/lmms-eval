import io
import json
import os
import pathlib
import re
import time
import tempfile
from typing import List, Tuple

import datasets
import numpy as np
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from google import genai
    from google.genai import types

    NUM_SECONDS_TO_SLEEP = 30
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

except Exception as e:
    eval_logger.error(f"Error importing google.genai: {str(e)}")
    genai = None

try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("Error importing decord, video frame sampling will not work")
    pass

try:
    import soundfile as sf
except Exception as e:
    eval_logger.warning(f"Error importing soundfile, audio generation will not work: {str(e)}")


@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-1.5-pro",
        # modality: str = "image",
        max_frames_num: int = 10,
        timeout: int = 120,
        continual_mode: bool = True,
        response_persistent_folder: str = "./logs/gemini_persistent_folder",
        interleave: bool = False,
        use_video_frames: bool = False,  # If True, extract frames (loses audio); If False, upload entire video (preserves audio)
        # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.max_frames_num = max_frames_num
        self.timeout = timeout
        self.continual_mode = continual_mode
        self.response_persistent_file = ""
        self.interleave = interleave
        self.use_video_frames = use_video_frames
        
        if genai:
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            self.client = None
            eval_logger.error("Google GenAI SDK not initialized")

        if self.continual_mode:
            self.response_persistent_folder = response_persistent_folder
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

        if os.path.exists(self.response_persistent_file):
            with open(self.response_persistent_file, "r") as f:
                self.response_cache = json.load(f)
            self.cache_mode = "resume"
        else:
            self.response_cache = {}
            self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        # self.modality = modality

        self.video_pool = []

    def free_video(self):
        # TODO: Implement deletion if needed for managing storage limits, though usually handled by TTL
        # For uploaded files, we might want to delete them.
        # self.client.files.delete(name=video.name)
        # But here video_pool stores the file objects/responses.
        self.video_pool = []

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def encode_video(self, video_path):
        uploaded_obj = self.client.files.upload(file=video_path)
        # Wait for processing
        while uploaded_obj.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_obj = self.client.files.get(name=uploaded_obj.name)
            
        self.video_pool.append(uploaded_obj)
        return uploaded_obj

    def encode_video_with_frames(self, video_path, max_frames_num):
        """
        Extract frames from video and upload them as individual images to Gemini API
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)

            # Ensure the last frame is included
            if total_frame_num - 1 not in uniform_sampled_frames:
                uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

            frame_idx = uniform_sampled_frames.tolist()
            frames = vr.get_batch(frame_idx).asnumpy()

            frame_images = []
            for frame in frames:
                img = Image.fromarray(frame)
                frame_images.append(img)

            return frame_images
        except Exception as e:
            eval_logger.error(f"Error extracting frames from video {video_path}: {str(e)}")
            # Fallback to original video upload method
            return [self.encode_video(video_path)]

    def encode_audio(self, audio):
        # Save to temp file because client.files.upload usually expects a path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio["array"], audio["sampling_rate"], format="WAV")
            tmp_path = tmp.name
        
        try:
            uploaded_obj = self.client.files.upload(file=tmp_path)
            return uploaded_obj
        finally:
            os.unlink(tmp_path)

    def convert_modality(self, images):
        converted_images = []
        for img in images:
            if isinstance(img, dict) and "sampling_rate" in img:  # audio
                try:
                    audio = self.encode_audio(img)
                    converted_images.append(audio)
                except Exception as e:
                    eval_logger.error(f"Error converting audio: {str(e)}")
            elif isinstance(img, str):  # video file path
                try:
                    # Check if it's a video file by extension
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
                    if any(img.lower().endswith(ext) for ext in video_extensions):
                        if self.use_video_frames:
                            # Extract frames from video (loses audio)
                            frame_images = self.encode_video_with_frames(img, self.max_frames_num)
                            converted_images.extend(frame_images)
                        else:
                            # Upload entire video with audio preserved
                            uploaded_obj = self.encode_video(img)
                            converted_images.append(uploaded_obj)
                    else:
                        # Treat as regular file upload
                        uploaded_obj = self.encode_video(img)
                        converted_images.append(uploaded_obj)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
            else:
                # Keep other types as-is (PIL Images, etc.)
                converted_images.append(img)
        return converted_images

    def construct_interleaved_input(self, content, media):
        pattern = r"<media_(\d+)>"
        parts = re.split(pattern, content)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part == "":
                    continue
                result.append(part)
            else:
                result.append(media[int(part)])

        return result

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self._rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            # Update GenerationConfig to new SDK format
            config = types.GenerateContentConfig(
                max_output_tokens=16384,
                temperature=gen_kwargs["temperature"],
                safety_settings=[
                    types.SafetySetting(
                        category='HARM_CATEGORY_DANGEROUS_CONTENT',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_HATE_SPEECH',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_HARASSMENT',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        threshold='BLOCK_NONE'
                    ),
                ]
            )

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            visuals = self.convert_modality(visuals)

            if self.interleave:
                message = self.construct_interleaved_input(contexts, visuals)
            else:
                message = [contexts] + visuals

            for attempt in range(5):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_version,
                        contents=message,
                        config=config,
                    )
                    
                    # Check if response has valid text content by safely trying to access it
                    try:
                        if response.text:
                            content = response.text
                            break
                    except (ValueError, AttributeError) as text_error:
                        # Response has no valid text content
                        eval_logger.info(f"No valid text content: {str(text_error)}")
                        
                    # Response was blocked or has no text content
                    # New SDK might not have prompt_feedback or candidates in same way, 
                    # but candidates usually exist.
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            eval_logger.info(f"Response blocked/incomplete, finish_reason: {candidate.finish_reason}")
                    content = ""
                    break
                        
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 3 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP * (attempt + 1))
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
            res.append(content)
            pbar.update(1)

            self.free_video()

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini API")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"

    def get_image_audio_text_interleaved_messsage(self, image_path, audio_path, question):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed image token and no audio in text
        for index in range(1, 1 + len(image_path)):
            question = question.replace(f"[img{index}]", "<image>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        image_counter = 0
        audio_counter = 0
        for part in re.split(r"(<image>|<audio>)", text):
            if part == "<image>":
                info_list.append(Image.open(image_path[image_counter]))
                image_counter += 1
            elif part == "<audio>":
                # For inline audio, we could use types.Part or upload. 
                # Original code used mime_type and data (bytes).
                # If using client.models.generate_content, we can pass types.Part.
                
                audio_bytes = pathlib.Path(audio_path[audio_counter]).read_bytes()
                # Construct a Part object for inline audio
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                info_list.append(audio_part)
                
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list

    def get_video_audio_text_interleaved_message(self, video_path, audio_path, question):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed video token and no audio in text
        for index in range(1, 1 + len(video_path)):
            question = question.replace(f"[video{index}]", "<video>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        video_counter = 0
        audio_counter = 0
        for part in re.split(r"(<video>|<audio>)", text):
            if part == "<video>":
                current_video_file_name = video_path[video_counter]
                # New SDK upload
                current_video_file = self.client.files.upload(file=current_video_file_name)
                
                # Wait for processing
                while current_video_file.state.name == "PROCESSING":
                    print("uploading file")
                    time.sleep(5)
                    current_video_file = self.client.files.get(name=current_video_file.name)
                
                if current_video_file.state.name == "FAILED":
                    print("uploading file failed, next question")
                    return 0
                info_list.append(current_video_file)
                video_counter += 1
            elif part == "<audio>":
                audio_bytes = pathlib.Path(audio_path[audio_counter]).read_bytes()
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                info_list.append(audio_part)
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list
