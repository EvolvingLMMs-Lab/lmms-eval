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
