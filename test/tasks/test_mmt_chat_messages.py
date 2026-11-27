"""MMT chat input must not keep the task-added group ``<image>`` marker next to the frame items.

``mmt_doc_to_text`` prefixes every MMT question with ``Question: <image>\n``. Chat processors add their own
image token per structured image item, so a four-frame sample reached the processor as five markers for four
images ("Found total 5 <image> tokens and 4 images"). The task-level ``mmt_doc_to_messages`` expands that
group marker into one image item per frame, in dataset order; the simple-model path (``mmt_doc_to_text``)
is unchanged. Frames are told apart by pixel fingerprints because ``mmt_doc_to_visual`` re-encodes them.
"""

import pytest
from PIL import Image

from lmms_eval.protocol import ChatMessages
from lmms_eval.tasks.mmt import utils

PROMPT_OPTIONS = {"pre_prompt": "", "post_prompt": "\nAnswer the question using a single character from the given options."}
FRAME_COLORS = ["red", "green", "blue", "yellow"]


def frame(color):
    return Image.new("RGB", (4, 4), color)


def doc(n_images=4, question="What happens across the frames?"):
    d = {"index": 0, "question": question, "A": "first option", "B": "second option", "answer": "A"}
    d["image"] = [frame(color) for color in FRAME_COLORS[:n_images]]
    return d


def messages_for(d):
    return utils.mmt_doc_to_messages(d, PROMPT_OPTIONS)


def content_of(messages_list):
    return messages_list[0]["content"]


def image_pixels(messages_list):
    return [item["url"].getpixel((0, 0)) for item in content_of(messages_list) if item["type"] == "image"]


def text_items(messages_list):
    return [item["text"] for item in content_of(messages_list) if item["type"] == "text"]


def test_four_frame_sample_expands_the_group_marker_in_dataset_order():
    d = doc(n_images=4)
    m = messages_for(d)
    assert image_pixels(m) == [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0)]
    assert all("<image>" not in text for text in text_items(m))
    joined = "\n".join(text_items(m))
    assert "Question:" in joined and d["question"] in joined
    assert "A: first option" in joined and PROMPT_OPTIONS["post_prompt"].strip() in joined
    # The protocol round trip keeps the frames in dataset order and no literal marker survives.
    chat = ChatMessages(messages=m)
    images, videos, audios = chat.extract_media()
    assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0)]
    assert videos == [] and audios == []
    hf = chat.to_hf_messages()
    assert sum(item["type"] == "image" for message in hf for item in message["content"]) == 4
    assert all("<image>" not in item["text"] for message in hf for item in message["content"] if item["type"] == "text")


def test_single_image_sample_keeps_one_image_item_and_full_prompt():
    d = doc(n_images=1)
    m = messages_for(d)
    assert image_pixels(m) == [(255, 0, 0)]
    joined = "\n".join(text_items(m))
    assert all("<image>" not in text for text in text_items(m))
    assert d["question"] in joined and "B: second option" in joined


def test_frame_reordering_is_detected_while_frame_counts_still_match():
    d = doc(n_images=4)
    swapped = doc(n_images=4)
    swapped["image"][2], swapped["image"][3] = swapped["image"][3], swapped["image"][2]
    expected, actual = image_pixels(messages_for(d)), image_pixels(messages_for(swapped))
    # A count-only check sees the same shape for both samples...
    assert len(expected) == len(actual) == 4
    # ...while the fingerprint check catches frames 2 and 3 being swapped, at the first divergence.
    assert actual[:2] == expected[:2] and actual[2:] != expected[2:]


def test_source_markers_are_preserved_only_the_task_added_prefix_is_expanded():
    d = doc(n_images=1, question="Which <image> matches the description?")
    m = messages_for(d)
    assert len(image_pixels(m)) == 1
    assert any("<image>" in text for text in text_items(m))


def test_pre_prompt_marker_does_not_move_the_frames():
    options = {**PROMPT_OPTIONS, "pre_prompt": "Look at <image> carefully.\n"}
    m = utils.mmt_doc_to_messages(doc(n_images=4), options)
    content = content_of(m)
    assert [item["type"] for item in content] == ["text", "image", "image", "image", "image", "text"]
    assert content[0]["text"] == "Look at <image> carefully.\nQuestion:"
    assert "<image>" not in content[-1]["text"]


def test_task_yaml_configs_wire_the_chat_hook():
    from pathlib import Path

    from lmms_eval.utils import load_yaml_config

    task_dir = Path(utils.__file__).parent
    for name in ("mmt_val", "mmt_test", "mmt_mi_val", "mmt_mi_test"):
        config = load_yaml_config(str(task_dir / f"{name}.yaml"))
        assert config["doc_to_messages"].__name__ == "mmt_doc_to_messages", name


def test_simple_model_prompt_is_unchanged():
    multi = utils.mmt_doc_to_text(doc(n_images=4), PROMPT_OPTIONS)
    single = utils.mmt_doc_to_text(doc(n_images=1), PROMPT_OPTIONS)
    for text in (multi, single):
        assert text.startswith("Question: <image>\n")
        assert text.count("<image>") == 1
        assert "A: first option" in text and PROMPT_OPTIONS["post_prompt"].strip() in text


def test_chat_template_renders_one_marker_per_frame():
    """Real-processor check for the marker invariant; skipped when the processor is not reachable."""
    import os

    transformers = pytest.importorskip("transformers")
    local_dir = os.environ.get("SMOLVLM_PROCESSOR_DIR")
    try:
        if local_dir:
            processor = transformers.AutoProcessor.from_pretrained(local_dir)
        else:
            processor = transformers.AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", revision="d039ab0d4b9311b617e8ffec1d06c8464364aeb3")
    except Exception as error:  # no network in CI
        pytest.skip(f"SmolVLM processor unavailable: {error}")
    for n_images in (1, 4):
        prompt = processor.apply_chat_template(ChatMessages(messages=messages_for(doc(n_images))).to_hf_messages(), tokenize=False, add_generation_prompt=True)
        assert prompt.count("<image>") == n_images
