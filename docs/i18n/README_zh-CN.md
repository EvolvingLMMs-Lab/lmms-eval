<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# 大型多模态模型评估套件

🌐 [English](../../README.md) | **简体中文** | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> 使用 `lmms-eval` 加速大型多模态模型（LMM）的开发与评估，支持文本、图像、视频、音频等多种任务。

🏠 [LMMs-Lab 主页](https://www.lmms-lab.com/) | 🤗 [Huggingface 数据集](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [支持的任务 (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [支持的模型 (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [文档](../README.md)

---

## 公告

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** 发布！此主要版本引入了全面的音频评估、响应缓存、5个新模型（GPT-4o Audio Preview、Gemma-3、LongViLA-R1、LLaVA-OneVision 1.5、Thyme）以及50多个新基准变体，涵盖音频（Step2、VoiceBench、WenetSpeech）、视觉（CharXiv、Lemonade）和推理（CSBench、SciBench、MedQA、SuperGPQA）等可复现结果。详情请参阅[发布说明](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md)。
- [2025-07] 🚀🚀 我们发布了 `lmms-eval-0.4`。详情请参阅[发布说明](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)。

## 为什么选择 `lmms-eval`？

我们正踏上通往通用人工智能（AGI）的征程，这份热情不亚于 1960 年代的登月计划。推动这一进程的是大型语言模型（LLM）和大型多模态模型（LMM），它们能够理解、学习并完成各类人类任务。

为了评估这些模型的能力，我们需要各种基准测试。然而现实是，这些基准和数据集散落在 Google Drive、Dropbox、各高校和实验室的网站上，找起来就像寻宝一样费劲。

在语言模型领域，[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 开创了先河。我们借鉴了它优雅高效的设计理念，打造了 **lmms-eval**，一个专为多模态模型设计的统一评估框架。

## 安装

### 使用 uv（推荐）

我们使用 `uv` 进行包管理，确保所有开发者的环境一致。首先安装 uv：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

克隆仓库并安装：
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# 推荐
uv pip install -e ".[all]"
# 如果您想使用 uv sync
# uv sync  # 这会从 uv.lock 创建/更新您的环境
```

执行命令：
```bash
uv run python -m lmms_eval --help
```

### 替代安装方式

直接从 Git 安装：
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# 使用此方式安装时，可能需要自行添加任务配置文件
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## 使用方法

> 更多示例请参见 [examples/models](../../examples/models)

**OpenAI 兼容模型的评估**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM 的评估**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision 的评估**

```bash
bash examples/models/llava_onevision.sh
```

**更多参数**

```bash
python3 -m lmms_eval --help
```

## 添加自定义模型和数据集

请参阅我们的[文档](../README.md)。

## 致谢

lmms-eval 基于 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 开发。建议阅读其[文档](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)了解更多背景。

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
