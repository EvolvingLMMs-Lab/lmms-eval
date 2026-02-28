"""Tests for ChatMessages protocol and content types.

Validates the unified message protocol that all chat models consume,
including construction, media extraction, and HuggingFace format conversion.
"""

import pytest
from PIL import Image
from pydantic import ValidationError

from lmms_eval.protocol import (
    ChatAudioContent,
    ChatImageContent,
    ChatMessage,
    ChatMessages,
    ChatTextContent,
    ChatVideoContent,
)


@pytest.fixture
def sample_image():
    return Image.new("RGB", (10, 10))


def _build_chat_messages(raw_messages):
    return ChatMessages.model_validate({"messages": raw_messages})


# ============================================================================
# ChatTextContent Tests
# ============================================================================


def test_chat_text_content_default_type():
    # Arrange
    content = ChatTextContent(text="hello")

    # Act & Assert
    assert content.type == "text"
    assert content.text == "hello"


def test_chat_text_content_model_validate_from_dict():
    # Arrange & Act
    content = ChatTextContent.model_validate({"text": "from dict"})

    # Assert
    assert content.type == "text"
    assert content.text == "from dict"


# ============================================================================
# ChatImageContent Tests
# ============================================================================


def test_chat_image_content_default_type(sample_image):
    # Arrange
    content = ChatImageContent(url=sample_image)

    # Act & Assert
    assert content.type == "image"
    assert isinstance(content.url, Image.Image)
    assert content.url.size == (10, 10)


def test_chat_image_content_model_validate_from_dict(sample_image):
    # Arrange & Act
    content = ChatImageContent.model_validate({"url": sample_image})

    # Assert
    assert content.type == "image"
    assert isinstance(content.url, Image.Image)
    assert content.url.size == (10, 10)


# ============================================================================
# ChatVideoContent Tests
# ============================================================================


def test_chat_video_content_default_type():
    # Arrange
    content = ChatVideoContent(url="video://sample.mp4")

    # Act & Assert
    assert content.type == "video"
    assert content.url == "video://sample.mp4"


def test_chat_video_content_model_validate_from_dict():
    # Arrange & Act
    content = ChatVideoContent.model_validate({"url": "video://dict.mp4"})

    # Assert
    assert content.type == "video"
    assert content.url == "video://dict.mp4"


# ============================================================================
# ChatAudioContent Tests
# ============================================================================


def test_chat_audio_content_default_type():
    # Arrange
    content = ChatAudioContent(url="audio://sample.wav")

    # Act & Assert
    assert content.type == "audio"
    assert content.url == "audio://sample.wav"


def test_chat_audio_content_model_validate_from_dict():
    # Arrange & Act
    content = ChatAudioContent.model_validate({"url": "audio://dict.wav"})

    # Assert
    assert content.type == "audio"
    assert content.url == "audio://dict.wav"


# ============================================================================
# ChatMessage Tests
# ============================================================================


@pytest.mark.parametrize("role", ["user", "system", "assistant"])
def test_chat_message_accepts_valid_roles(role):
    # Arrange & Act
    message = ChatMessage(role=role, content=[ChatTextContent(text="ok")])

    # Assert
    assert message.role == role
    assert len(message.content) == 1
    assert isinstance(message.content[0], ChatTextContent)


def test_chat_message_rejects_invalid_role():
    # Arrange & Act & Assert
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content=[ChatTextContent(text="no")])


def test_chat_message_supports_mixed_content():
    # Arrange & Act
    message = ChatMessage(
        role="user",
        content=[
            ChatTextContent(text="hello"),
            ChatImageContent(url="img://1"),
            ChatVideoContent(url="vid://1"),
            ChatAudioContent(url="aud://1"),
        ],
    )

    # Assert
    assert [item.type for item in message.content] == ["text", "image", "video", "audio"]


def test_chat_message_model_validate_mixed_content_dict():
    # Arrange & Act
    message = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "img://1"},
                {"type": "video", "url": "vid://1"},
                {"type": "audio", "url": "aud://1"},
            ],
        }
    )

    # Assert
    assert isinstance(message.content[0], ChatTextContent)
    assert isinstance(message.content[1], ChatImageContent)
    assert isinstance(message.content[2], ChatVideoContent)
    assert isinstance(message.content[3], ChatAudioContent)


# ============================================================================
# ChatMessages Tests
# ============================================================================


def test_chat_messages_model_validate_from_raw_dict_list():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {"role": "system", "content": [{"text": "policy"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "img://a"},
                    {"text": "describe this"},
                ],
            },
        ]
    )

    # Assert
    assert len(messages.messages) == 2
    assert messages.messages[0].content[0].type == "text"
    assert messages.messages[1].content[0].type == "image"
    assert messages.messages[1].content[1].type == "text"


def test_chat_messages_preserves_message_order():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {"role": "system", "content": [{"text": "s"}]},
            {"role": "user", "content": [{"text": "u"}]},
            {"role": "assistant", "content": [{"text": "a"}]},
        ]
    )

    # Assert
    assert [message.role for message in messages.messages] == ["system", "user", "assistant"]


# ============================================================================
# extract_media() Tests
# ============================================================================


def test_extract_media_empty_messages():
    # Arrange
    messages = ChatMessages(messages=[])

    # Act
    images, videos, audios = messages.extract_media()

    # Assert
    assert images == []
    assert videos == []
    assert audios == []


def test_extract_media_text_only_messages():
    # Arrange
    messages = ChatMessages(messages=[ChatMessage(role="user", content=[ChatTextContent(text="hello")])])

    # Act
    images, videos, audios = messages.extract_media()

    # Assert
    assert images == []
    assert videos == []
    assert audios == []


def test_extract_media_images_only_messages():
    # Arrange
    red = Image.new("RGB", (10, 10), color="red")
    blue = Image.new("RGB", (10, 10), color="blue")
    messages = ChatMessages(
        messages=[
            ChatMessage(
                role="user",
                content=[ChatImageContent(url=red), ChatImageContent(url=blue)],
            )
        ]
    )

    # Act
    images, videos, audios = messages.extract_media()

    # Assert
    assert len(images) == 2
    assert all(isinstance(image, Image.Image) for image in images)
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[1].getpixel((0, 0)) == (0, 0, 255)
    assert videos == []
    assert audios == []


def test_extract_media_videos_only_messages():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": "video://one.mp4"},
                    {"type": "video", "url": "video://two.mp4"},
                ],
            }
        ]
    )

    # Act
    images, videos, audios = messages.extract_media()

    # Assert
    assert images == []
    assert videos == ["video://one.mp4", "video://two.mp4"]
    assert audios == []


def test_extract_media_audios_only_messages():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "audio", "url": "audio://one.wav"},
                    {"type": "audio", "url": "audio://two.wav"},
                ],
            }
        ]
    )

    # Act
    images, videos, audios = messages.extract_media()

    # Assert
    assert images == []
    assert videos == []
    assert audios == ["audio://one.wav", "audio://two.wav"]


def test_extract_media_mixed_media_types():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"text": "look"},
                    {"type": "image", "url": "img://one"},
                    {"type": "video", "url": "video://one"},
                    {"type": "audio", "url": "audio://one"},
                    {"type": "image", "url": "img://two"},
                ],
            }
        ]
    )

    # Act
    images, videos, audios = messages.extract_media()

    # Assert
    assert images == ["img://one", "img://two"]
    assert videos == ["video://one"]
    assert audios == ["audio://one"]


def test_extract_media_multiple_messages_multiple_items():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "img://u1"},
                    {"type": "video", "url": "video://u1"},
                    {"type": "audio", "url": "audio://u1"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "video", "url": "video://a1"},
                    {"type": "image", "url": "img://a1"},
                    {"type": "audio", "url": "audio://a1"},
                ],
            },
            {"role": "system", "content": [{"text": "meta"}]},
        ]
    )

    # Act
    images, videos, audios = messages.extract_media()

    # Assert
    assert images == ["img://u1", "img://a1"]
    assert videos == ["video://u1", "video://a1"]
    assert audios == ["audio://u1", "audio://a1"]


# ============================================================================
# to_hf_messages() Tests
# ============================================================================


def test_to_hf_messages_text_only():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {"role": "user", "content": [{"text": "hello"}]},
        ]
    )

    # Act
    hf_messages = messages.to_hf_messages()

    # Assert
    assert hf_messages == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]


def test_to_hf_messages_image_content_structure():
    # Arrange
    image_url = "https://example.com/image.png"
    messages = _build_chat_messages(
        [
            {"role": "user", "content": [{"type": "image", "url": image_url}]},
        ]
    )

    # Act
    hf_messages = messages.to_hf_messages()
    image_content = hf_messages[0]["content"][0]

    # Assert
    assert image_content["type"] == "image"
    assert image_content["image"] == image_url


def test_to_hf_messages_video_content_with_kwargs():
    # Arrange
    video_kwargs = {"nframes": 8, "fps": "1.0", "backend": "decord"}
    messages = _build_chat_messages(
        [
            {"role": "user", "content": [{"type": "video", "url": "video://clip.mp4"}]},
        ]
    )

    # Act
    hf_messages = messages.to_hf_messages(video_kwargs=video_kwargs)
    video_content = hf_messages[0]["content"][0]

    # Assert
    assert video_content["type"] == "video"
    assert video_content["video"] == "video://clip.mp4"
    for key, value in video_kwargs.items():
        assert video_content[key] == value


def test_to_hf_messages_video_content_without_kwargs():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {"role": "user", "content": [{"type": "video", "url": "video://clip.mp4"}]},
        ]
    )

    # Act
    hf_messages = messages.to_hf_messages(video_kwargs=None)

    # Assert
    assert hf_messages[0]["content"][0] == {"type": "video", "video": "video://clip.mp4"}


def test_to_hf_messages_audio_content_structure():
    # Arrange
    audio_url = "audio://clip.wav"
    messages = _build_chat_messages(
        [
            {"role": "assistant", "content": [{"type": "audio", "url": audio_url}]},
        ]
    )

    # Act
    hf_messages = messages.to_hf_messages()
    audio_content = hf_messages[0]["content"][0]

    # Assert
    assert audio_content["type"] == "audio"
    assert audio_content["audio"] == audio_url


def test_to_hf_messages_mixed_content_in_single_message():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"text": "inspect"},
                    {"type": "image", "url": "img://1"},
                    {"type": "video", "url": "video://1"},
                    {"type": "audio", "url": "audio://1"},
                ],
            }
        ]
    )

    # Act
    hf_messages = messages.to_hf_messages(video_kwargs={"nframes": 4})

    # Assert
    assert hf_messages[0]["content"][0] == {"type": "text", "text": "inspect"}
    assert hf_messages[0]["content"][1] == {"type": "image", "image": "img://1"}
    assert hf_messages[0]["content"][2] == {"type": "video", "video": "video://1", "nframes": 4}
    assert hf_messages[0]["content"][3] == {"type": "audio", "audio": "audio://1"}


def test_to_hf_messages_video_kwargs_none_equals_empty_dict():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {"role": "user", "content": [{"type": "video", "url": "video://same.mp4"}]},
        ]
    )

    # Act
    hf_none = messages.to_hf_messages(video_kwargs=None)
    hf_empty = messages.to_hf_messages(video_kwargs={})

    # Assert
    assert hf_none == hf_empty


def test_to_hf_messages_multiple_messages_roles_preserved():
    # Arrange & Act
    messages = _build_chat_messages(
        [
            {"role": "system", "content": [{"text": "rules"}]},
            {"role": "user", "content": [{"type": "image", "url": "img://1"}, {"text": "question"}]},
            {"role": "assistant", "content": [{"text": "answer"}]},
        ]
    )

    # Act
    hf_messages = messages.to_hf_messages()

    # Assert
    assert [message["role"] for message in hf_messages] == ["system", "user", "assistant"]
    assert hf_messages[1]["content"][0] == {"type": "image", "image": "img://1"}
    assert hf_messages[1]["content"][1] == {"type": "text", "text": "question"}


def test_to_hf_messages_does_not_mutate_video_kwargs():
    # Arrange
    video_kwargs = {"nframes": 16, "backend": "decord"}
    messages = _build_chat_messages(
        [
            {"role": "user", "content": [{"type": "video", "url": "video://clip.mp4"}]},
        ]
    )

    # Act
    _ = messages.to_hf_messages(video_kwargs=video_kwargs)

    # Assert
    assert video_kwargs == {"nframes": 16, "backend": "decord"}
