"""The auto ``doc_to_messages`` path must not keep literal ``<image>`` markers next to the images.

Tasks such as STARE, MMT or Geometry3K write ``<image>`` into ``doc_to_text`` for simple models. Chat processors add
their own image token per image, so with the markers kept in the text a chat model saw every image twice
("Found [4] <image> tokens and [2] images"). When the markers match the images, each image goes at its marker.
"""

from PIL import Image

from lmms_eval.api.task import ConfigurableMessagesTask

RED = Image.new("RGB", (4, 4), "red")
BLUE = Image.new("RGB", (4, 4), "blue")


def _content(visuals, text):
    return ConfigurableMessagesTask._build_user_turn(visuals, text)[0]["content"]


def test_images_are_placed_at_their_markers():
    content = _content([RED, BLUE], "Compare <image> with <image>\nAnswer A or B.")
    assert [item["type"] for item in content] == ["text", "image", "text", "image", "text"]
    assert content[1]["url"] is RED and content[3]["url"] is BLUE
    assert [item["text"] for item in content if item["type"] == "text"] == ["Compare", "with", "Answer A or B."]


def test_marker_only_text_keeps_no_empty_text_items():
    content = _content([RED], "<image>")
    assert content == [{"type": "image", "url": RED}]


def test_non_image_media_stay_first():
    content = _content(["clip.mp4", RED], "Watch the clip. Then look at <image> and answer.")
    assert [item["type"] for item in content] == ["video", "text", "image", "text"]


def test_text_without_markers_is_unchanged():
    content = _content([RED, BLUE], "Describe both images.")
    assert [item["type"] for item in content] == ["image", "image", "text"]
    assert content[-1]["text"] == "Describe both images."


def test_marker_count_mismatch_is_unchanged():
    text = "Look at <image> only once."
    content = _content([RED, BLUE], text)
    assert [item["type"] for item in content] == ["image", "image", "text"]
    assert content[-1]["text"] == text
