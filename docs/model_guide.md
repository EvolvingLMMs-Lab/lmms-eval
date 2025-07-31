# New Model Guide
In order to properly evaluate a given LM, we require implementation of a wrapper class subclassing the `lmms_eval.api.model.lmms` class, that defines how the lmms_eval should interface with your model. This guide walks through how to write this `lmms` subclass via adding it to the library!


## Model Type (New Features)
- Chat (recommended) - The new recommended model type to be used for the future. Better support for interleaved text, image, video, and audio for multi-modal domain. Should be the type of the model to be implemented for new integrated models
- Simple (Legacy) - The original and legacy models that use `doc_to_visual` and `doc_to_text` to control the input of the model. You can still add in new models that belongs to this category.


## Setup

To get started contributing, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/<YOUR-USERNAME>/lmms-eval.git
cd lmms-eval
git checkout -b <model-type>
pip install -e .
```

Now, we'll create a new file where we'll be adding our model:

```sh
# (recommended) For chat models
touch lmms_eval/models/chat/<my_model_filename>.py

# For legacy simple models
touch lmms_eval/models/simple/<my_model_filename>.py
```

**As a rule of thumb, we recommend you to use `lmms_eval/models/chat/qwen_vl.py` and `lmms_eval/models/simple/instructblip.py` as reference implementations for your model. You can copy and paste the contents of one of these files into your new file to get started.**

## Interface

All models must subclass the `lmms_eval.api.model.lmms` class.

The lmms class enforces a common interface via which we can extract responses from a model:

```python
class MyCustomLM(lmms):
    #...
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        #...

    def generate_until(self, requests: list[Instance]) -> list[str]:
        #...
    #...
```
Where `Instance` is a dataclass defined in [`lmms_eval.api.instance`](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/api/instance.py) with property `args` of request-dependent type signature described below.

We support three types of requests, consisting of different interactions / measurements with an autoregressive LM.

All three request types take as input `requests` of type `list[Instance]` that have a matching `Instance.request_type` to the method name. Overall, you can check the [construct_requests](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/api/task.py#L918) to see how the arguments are being constructed for different types of output type requests.

- `generate_until`
  - Each request contains `Instance.args : Tuple[str, dict]` containing 1. an input string to the LM and 2. a dictionary of keyword arguments used to control generation parameters.
  - In each `Instance.args`,
    - Chat Model : there will be 5 elements which are `doc_to_messages, all_gen_kwargs, doc_id, task, split`. `doc_to_messages` refers to the messages input for the LMM. Sometimes it might contains image token and need to address differently for different models. `all_gen_kwargs` refers to the dict that contains all the generation configuration for the model. We use `doc_id`, `task`, and `split` to access the dataset and then you can use `doc_to_messages` which is a function reference to process the input into interleaved format. When you implement your own model, you should use these to write your own generate_util function.
    - Simple Model (Legacy) : there will be 6 elements which are `contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split`. `contexts` refers to the formatted question and is the text input for the LMM. Sometimes it might contains image token and need to address differently for different models. `all_gen_kwargs` refers to the dict that contains all the generation configuration for the model. We use `doc_id`, `task`, and `split` to access the dataset and then you can use `doc_to_visual` which is a function reference to process the image. When you implement your own model, you should use these to write your own generate_util function.
  - Using this input and these generation parameters, text will be sampled from the language model (typically until a maximum output length or specific stopping string sequences--for example, `{"until": ["\n\n", "."], "max_new_tokens": 128}`).
  - The generated input+output text from the model will then be returned.

- `loglikelihood`
  - Each request contains `Instance.args : Tuple[str, str]` containing 1. an input string to the LM and 2. a target string on which the loglikelihood of the LM producing this target, conditioned on the input, will be returned.
  - In each `Instance.args` there will be 6 elements which are ` contexts, doc_to_target, doc_to_visual, doc_id, task, split`. `contexts` refers to the formatted question and is the text input for the LMM. Sometimes it might contains image token and need to address differently for different models. `doc_to_target` is a function reference that get the get the answer from the doc. This will be the continuation of the answer and only tokens belong to this part should be calculated for the loglikelihood.
  - Each request will have, as result, `(ll, is_greedy): Tuple[float, int]` returned, where `ll` is a floating point number representing the log probability of generating the target string conditioned on the input, and `is_greedy` being either the value `0` or `1`, with it being `1` if and only if the target string *would be generated by greedy sampling from the LM* (that is, if the  target string is the *most likely* N-token string to be output by the LM given the input. )




## Registration

Congrats on implementing your model! Now it's time to test it out.

To make your model usable via the command line interface to `lmms_eval`, you'll need to tell `lmms_eval` what your model's name is.

This is done via a *decorator*, `lmms_eval.api.registry.register_model`. Using `register_model()`, one can both tell the package what the model's name(s) to be used are when invoking it with `python -m lmms-eval --model <name>` and alert `lmms_eval` to the model's existence.

```python
from lmms_eval.api.registry import register_model

@register_model("<name1>", "<name2>")
class MyCustomLM(LM):
    # is_simple = False for chat model
    # is_simple = True for simple model (default to True)
```

The final step is to import your model in `lmms_eval/models/__init__.py`:
```python
from .my_model_filename import MyCustomLM
```

## Complete Model Examples

### Image Model Example (Primary Use Case)

Here's a complete example of implementing an image model using the chat interface:

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from typing import List, Tuple
import torch
from PIL import Image

@register_model("my_image_model")
class MyImageModel(lmms):
    is_simple = False  # Use chat model type (recommended)
    
    def __init__(self, pretrained: str, device: str = "cuda", **kwargs):
        super().__init__()
        self.device = device
        # Initialize your vision-language model here
        self.model = load_your_model(pretrained)
        self.processor = load_your_processor(pretrained)
        
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            # Extract components from the request
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            
            # Get the document and process messages
            doc = self.task_dict[task][split][doc_id]
            messages = doc_to_messages(doc)
            
            # Process images and text from messages
            images, videos, audios = messages.extract_media()
            text_prompt = ""
            for message in messages:
                if message.type == "text":
                    text_prompt += message["text"]

            # If your model support apply chat template
            # text_prompt = self.processor.apply_chat_template(messages.to_hf_messages())
            
            # Prepare inputs for your model
            inputs = self.processor(
                text=text_prompt, 
                images=images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                    temperature=gen_kwargs.get("temperature", 0.0),
                    do_sample=gen_kwargs.get("do_sample", False)
                )
            
            # Decode and return response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            results.append(response)
            
        return results
    
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        # Implement loglikelihood computation for multiple choice tasks
        results = []
        for request in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = request.args
            # Implementation for computing log probabilities
            # ...
        return results
```

### Video Model Extension

For video models, extend the image model pattern to handle temporal data:

```python
@register_model("my_video_model")
class MyVideoModel(lmms):
    is_simple = False
    
    def __init__(self, pretrained: str, max_frames: int = 8, **kwargs):
        super().__init__()
        self.max_frames = max_frames
        # Initialize video model
        
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = doc_to_messages(doc)
            
            # Extract video frames
            images, videos, audios = messages.extract_media()
            text_prompt = ""
            for message in messages:
                if message.type == "text":
                    text_prompt += message["text"]
            
            # Process video frames and generate response
            # ...
        return results
    
    def extract_frames(self, video_path, max_frames):
        # Extract frames from video file
        # Return list of PIL Images or tensors
        pass
```

### Audio Model Extension

For audio models, adapt the pattern to handle audio inputs:

```python
@register_model("my_audio_model")
class MyAudioModel(lmms):
    is_simple = False
    
    def __init__(self, pretrained: str, sample_rate: int = 16000, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        # Initialize audio model
        
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = doc_to_messages(doc)
            
            images, videos, audios = messages.extract_media()
            text_prompt = ""
            for message in messages:
                if message.type == "text":
                    text_prompt += message["text"]
            
            # Process audio and generate response
            # ...
        return results
    
    def load_audio(self, audio_path, sample_rate):
        # Load audio file and resample if needed
        # Return audio tensor or array
        pass
```

## Key Implementation Notes

1. **Image Models**: Handle visual inputs through PIL Images or tensors, typically support single or multiple images
2. **Video Models**: Extract frames from videos, handle temporal relationships
3. **Audio Models**: Process audio waveforms, handle different sample rates and formats

Remember to:
- Handle different input modalities in the `doc_to_messages` function
- Process model-specific tokens (e.g., `<image>`, `<video>`, `<audio>`)
- Implement both `generate_until` and `loglikelihood` methods if your model supports both generation and multiple-choice tasks
- Follow the existing model implementations in `lmms_eval/models/chat/` for reference
