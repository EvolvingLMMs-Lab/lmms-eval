<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
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

📖 [Tác Vụ Được Hỗ Trợ (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [Mô Hình Được Hỗ Trợ (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Tài Liệu](../README.md)

---

## Thông Báo

Việc đánh giá các mô hình đa phương thức khó hơn chúng ta tưởng. Chúng ta có hàng trăm benchmark, nhưng không có cách tiêu chuẩn nào để chạy chúng. Kết quả khác nhau giữa các phòng thí nghiệm. Các so sánh trở nên không đáng tin cậy. Chúng tôi đã và đang nỗ lực giải quyết vấn đề này - không phải thông qua những nỗ lực phi thường, mà thông qua một quy trình có hệ thống.

**Tháng 1 năm 2026** - Chúng tôi nhận thấy rằng khả năng suy luận không gian và bố cục vẫn là những điểm mù trong các benchmark hiện tại. Chúng tôi đã thêm [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/), và [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Đối với các nhóm vận hành quy trình đánh giá từ xa, chúng tôi đã giới thiệu máy chủ đánh giá HTTP (#972). Đối với những người cần sự chặt chẽ về thống kê, chúng tôi đã thêm CLT và ước tính sai số chuẩn theo cụm (clustered standard error estimation) (#989).

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** đã ra mắt! Bản phát hành chính này giới thiệu đánh giá âm thanh toàn diện, bộ nhớ đệm phản hồi, 5 mô hình mới (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), và hơn 50 biến thể benchmark mới bao gồm âm thanh (Step2, VoiceBench, WenetSpeech), thị giác (CharXiv, Lemonade), và suy luận (CSBench, SciBench, MedQA, SuperGPQA). Xem [ghi chú phát hành](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md) để biết chi tiết.
- [2025-07] 🚀🚀 Chúng tôi đã phát hành `lmms-eval-0.4`. Xem [ghi chú phát hành](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.4.md) để biết thêm chi tiết.

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

**Đánh Giá LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Đánh Giá LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Đánh Giá Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Đánh Giá với tensor parallel cho mô hình lớn (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Đánh Giá với SGLang cho mô hình lớn (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Thêm Tham Số**

```bash
python3 -m lmms_eval --help
```

**Biến Môi Trường**
Trước khi chạy các thí nghiệm và đánh giá, chúng tôi khuyến nghị bạn xuất các biến môi trường sau vào môi trường của mình. Một số biến là cần thiết để một số tác vụ nhất định có thể chạy được.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Các biến môi trường khác có thể bao gồm
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY v.v.
```

**Các Vấn Đề Môi Trường Thường Gặp**

Đôi khi bạn có thể gặp phải một số vấn đề phổ biến, ví dụ như lỗi liên quan đến httpx hoặc protobuf. Để giải quyết các vấn đề này, trước tiên bạn có thể thử:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Nếu bạn đang sử dụng numpy==2.x, đôi khi có thể gây ra lỗi
python3 -m pip install numpy==1.26;
# Đôi khi sentencepiece là cần thiết để tokenizer hoạt động
python3 -m pip install sentencepiece;
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

@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}
```
