<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# 大型多模態模型評估套件

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | **繁體中文** | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> 使用 `lmms-eval` 加速大型多模態模型（LMM）的開發與評估，支援文字、影像、視訊、音訊等多種任務。

🏠 [LMMs-Lab 首頁](https://www.lmms-lab.com/) | 🤗 [Huggingface 資料集](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [支援的任務 (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [支援的模型 (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [文件](../README.md)

---

## 公告

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** 發布！此主要版本引入了全面的音訊評估、回應快取、5個新模型（GPT-4o Audio Preview、Gemma-3、LongViLA-R1、LLaVA-OneVision 1.5、Thyme）以及50多個新基準變體，涵蓋音訊（Step2、VoiceBench、WenetSpeech）、視覺（CharXiv、Lemonade）和推理（CSBench、SciBench、MedQA、SuperGPQA）等可重現結果。詳情請參閱[發布說明](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md)。
- [2025-07] 🚀🚀 我們發布了 `lmms-eval-0.4`。詳情請參閱[發布說明](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)。

## 為什麼選擇 `lmms-eval`？

我們正踏上通往通用人工智慧（AGI）的征程，這份熱情不亞於 1960 年代的登月計畫。推動這一進程的是大型語言模型（LLM）和大型多模態模型（LMM），它們能夠理解、學習並完成各類人類任務。

為了評估這些模型的能力，我們需要各種基準測試。然而現實是，這些基準和資料集散落在 Google Drive、Dropbox、各大學和實驗室的網站上，找起來就像尋寶一樣費勁。

在語言模型領域，[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 開創了先河。我們借鏡了它優雅高效的設計理念，打造了 **lmms-eval**，一個專為多模態模型設計的統一評估框架。

## 安裝

### 使用 uv（推薦）

我們使用 `uv` 進行套件管理，確保所有開發者的環境一致。首先安裝 uv：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

複製儲存庫並安裝：
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# 推薦
uv pip install -e ".[all]"
# 如果您想使用 uv sync
# uv sync  # 這會從 uv.lock 建立/更新您的環境
```

執行指令：
```bash
uv run python -m lmms_eval --help
```

### 替代安裝方式

直接從 Git 安裝：
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# 使用此方式安裝時，可能需要自行新增任務設定檔
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## 使用方法

> 更多範例請參見 [examples/models](../../examples/models)

**OpenAI 相容模型的評估**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM 的評估**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision 的評估**

```bash
bash examples/models/llava_onevision.sh
```

**更多參數**

```bash
python3 -m lmms_eval --help
```

## 新增自訂模型和資料集

請參閱我們的[文件](../README.md)。

## 致謝

lmms-eval 基於 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 開發。建議閱讀其[文件](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)了解更多背景。

## 引用

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
