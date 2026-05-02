from lmms_eval.models.chat.openai import _ctx_to_text_chat_messages


def test_ctx_replaces_auto_text_only_doc_message():
    ctx = "Task instructions.\n\nFew-shot example.\n\nQuestion: What is 2 + 2?"
    raw_messages = [{"role": "user", "content": [{"type": "text", "text": "Question: What is 2 + 2?"}]}]

    assert _ctx_to_text_chat_messages(ctx, raw_messages) == [{"role": "user", "content": [{"type": "text", "text": ctx}]}]


def test_ctx_does_not_replace_multimodal_messages():
    raw_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "chart.png"},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]

    assert _ctx_to_text_chat_messages("Use the full benchmark prompt", raw_messages) is raw_messages


def test_ctx_does_not_replace_explicit_multi_message_chat():
    raw_messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
        {"role": "user", "content": [{"type": "text", "text": "Question"}]},
    ]

    assert _ctx_to_text_chat_messages("Full prompt", raw_messages) is raw_messages
