<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Bộ Công Cụ Đánh Giá Mô Hình Đa Phương Thức Lớn

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | **Tiếng Việt** | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Tăng tốc phát triển các mô hình đa phương thức lớn (LMMs) với `lmms-eval`. Chúng tôi hỗ trợ hầu hết các tác vụ văn bản, hình ảnh, video và âm thanh.

🏠 [Trang Chủ LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Bộ Dữ Liệu Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Tác Vụ Được Hỗ Trợ (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Mô Hình Được Hỗ Trợ (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Tài Liệu](../README.md)

---

## Thông Báo

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** đã ra mắt! Bản phát hành chính này giới thiệu đánh giá âm thanh toàn diện, bộ nhớ đệm phản hồi, 5 mô hình mới (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), và hơn 50 biến thể benchmark mới bao gồm âm thanh (Step2, VoiceBench, WenetSpeech), thị giác (CharXiv, Lemonade), và suy luận (CSBench, SciBench, MedQA, SuperGPQA). Xem [ghi chú phát hành](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) để biết chi tiết.
- [2025-07] 🚀🚀 Chúng tôi đã phát hành `lmms-eval-0.4`. Xem [ghi chú phát hành](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) để biết thêm chi tiết.

## Tại Sao Chọn `lmms-eval`?

Chúng ta đang trong một hành trình thú vị hướng tới việc tạo ra Trí Tuệ Nhân Tạo Tổng Quát (AGI), tương tự như sự nhiệt tình của cuộc đổ bộ lên Mặt Trăng những năm 1960. Hành trình này được thúc đẩy bởi các mô hình ngôn ngữ lớn tiên tiến (LLMs) và các mô hình đa phương thức lớn (LMMs), là các hệ thống phức tạp có khả năng hiểu, học hỏi và thực hiện nhiều loại nhiệm vụ của con người.

Để đo lường mức độ tiên tiến của các mô hình này, chúng tôi sử dụng nhiều benchmark đánh giá khác nhau. Các benchmark này là công cụ giúp chúng tôi hiểu khả năng của các mô hình này, cho chúng tôi thấy chúng ta đang gần đến AGI như thế nào. Tuy nhiên, việc tìm kiếm và sử dụng các benchmark này là một thách thức lớn.

Trong lĩnh vực mô hình ngôn ngữ, công trình của [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) đã tạo tiền lệ quý báu. Chúng tôi đã tiếp thu thiết kế tinh tế và hiệu quả của lm-evaluation-harness và giới thiệu **lmms-eval**, một framework đánh giá được xây dựng tỉ mỉ để đánh giá LMM một cách nhất quán và hiệu quả.

## Cài Đặt

### Sử Dụng uv (Khuyến nghị cho môi trường nhất quán)

Chúng tôi sử dụng `uv` để quản lý gói nhằm đảm bảo tất cả các nhà phát triển sử dụng cùng phiên bản gói. Đầu tiên, cài đặt uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Để phát triển với môi trường nhất quán:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Khuyến nghị
uv pip install -e ".[all]"
# Nếu bạn muốn sử dụng uv sync
# uv sync  # Điều này tạo/cập nhật môi trường của bạn từ uv.lock
```

Để chạy lệnh:
```bash
uv run python -m lmms_eval --help  # Chạy bất kỳ lệnh nào với uv run
```

### Cài Đặt Thay Thế

Để sử dụng trực tiếp từ Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Bạn có thể cần thêm và bao gồm yaml tác vụ của riêng mình nếu sử dụng cài đặt này
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Cách Sử Dụng

> Xem thêm ví dụ tại [examples/models](../../examples/models)

**Đánh Giá Mô Hình Tương Thích OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Đánh Giá vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Đánh Giá LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Thêm Tham Số**

```bash
python3 -m lmms_eval --help
```

## Thêm Mô Hình và Bộ Dữ Liệu Tùy Chỉnh

Xem [tài liệu](../README.md) của chúng tôi.

## Lời Cảm Ơn

lmms_eval là một nhánh của [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Chúng tôi khuyến nghị đọc [tài liệu của lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) để biết thông tin liên quan.

## Trích Dẫn

```shell
@misc{zhang2024lmmsevalrealitycheckevaluation,
      title={LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models}, 
      author={Kaichen Zhang and Bo Li and Peiyuan Zhang and Fanyi Pu and Joshua Adrian Cahyono and Kairui Hu and Shuai Liu and Yuanhan Zhang and Jingkang Yang and Chunyuan Li and Ziwei Liu},
      year={2024},
      eprint={2407.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12772}, 
}
```
