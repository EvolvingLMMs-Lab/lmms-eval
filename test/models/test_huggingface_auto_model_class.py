from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from lmms_eval.models.chat.huggingface import resolve_auto_model_class


def test_vision_language_config_resolves_to_image_text_to_text():
    assert resolve_auto_model_class(AutoConfig.for_model("idefics3")) is AutoModelForImageTextToText
    assert resolve_auto_model_class(AutoConfig.for_model("qwen2_vl")) is AutoModelForImageTextToText


def test_text_only_config_resolves_to_causal_lm():
    assert resolve_auto_model_class(AutoConfig.for_model("llama")) is AutoModelForCausalLM


def test_config_without_generation_head_falls_back_to_auto_model():
    assert resolve_auto_model_class(AutoConfig.for_model("clip")) is AutoModel


def test_remote_code_config_uses_auto_map():
    config = AutoConfig.for_model("llama")

    class RemoteConfig(type(config)):
        pass

    remote = RemoteConfig()
    remote.auto_map = {"AutoConfig": "configuration_x.XConfig", "AutoModelForCausalLM": "modeling_x.XForCausalLM"}
    assert resolve_auto_model_class(remote) is AutoModelForCausalLM
