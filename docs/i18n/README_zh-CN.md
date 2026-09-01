<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
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

📖 [支持的任务 (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [支持的模型 (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [文档](../README.md)

---

## 更新动态

评估多模态模型比看起来要困难。我们有数百个基准测试，但没有运行它们的标准方法。不同实验室之间的结果各不相同，导致对比变得不可靠。我们一直在致力于解决这个问题——不是通过英雄式的个人努力，而是通过系统化的流程。

**2026年1月** - 我们意识到空间推理和组合推理仍然是现有基准测试的盲点。我们添加了 [CaptionQA](https://captionqa.github.io/)、[SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench)、[SiteBench](https://sitebench.github.io/) 和 [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial)。对于运行远程评估流水线的团队，我们引入了 HTTP 评估服务器 (#972)。对于需要统计严密性的用户，我们添加了中心极限定理（CLT）和聚类标准误差估计 (#989)。

**2025年10月 (v0.5)** - 音频评估曾是一个空白。模型可以“听”，但我们没有一致的方法来测试它们。此版本添加了全面的音频评估、响应缓存以提高效率，以及 50 多个涵盖音频、视觉和推理的基准变体。[发布说明](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md)。

<details>
<summary>以下是由我们优秀的贡献者添加的近期任务、模型和功能的按时间顺序排列的列表。</summary>

- [2025-01] 🎓🎓 我们发布了新的基准测试：[Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826)。详情请参阅[项目主页](https://videommmu.github.io/)。
- [2024-12] 🎉🎉 我们与 [MME 团队](https://github.com/BradyFU/Video-MME)和 [OpenCompass 团队](https://github.com/open-compass)共同发布了 [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296)。
- [2024-11] 🔈🔊 `lmms-eval/v0.3.0` 已升级，支持对 Qwen2-Audio 和 Gemini-Audio 等音频模型在 AIR-Bench、Clotho-AQA、LibriSpeech 等任务上进行音频评估。详情请参阅[博客](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.3.md)！
- [2024-10] 🎉🎉 欢迎新任务 [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench)，这是一个以视觉为核心的 VQA 基准测试 (NeurIPS'24)，通过关于自然图像的简单问题挑战视觉语言模型。
- [2024-10] 🎉🎉 欢迎新任务 [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench)，用于视频的细粒度时间理解和推理，揭示了巨大的 (>30%) 人机差距。
- [2024-10] 🎉🎉 欢迎新任务 [VDC](https://rese1f.github.io/aurora-web/)（用于视频详细字幕生成）、[MovieChat-1K](https://rese1f.github.io/MovieChat/)（用于长视频理解）和 [Vinoground](https://vinoground.github.io/)（一个由 1000 个短自然视频-字幕对组成的时间反事实 LMM 基准测试）。同时欢迎新模型：[AuroraCap](https://github.com/rese1f/aurora) 和 [MovieChat](https://github.com/rese1f/MovieChat)。
- [2024-09] 🎉🎉 欢迎新任务 [MMSearch](https://mmsearch.github.io/) 和 [MME-RealWorld](https://mme-realworld.github.io/) 以加速推理。
- [2024-09] ⚙️️⚙️️️️ 我们将 `lmms-eval` 升级到 `0.2.3`，增加了更多任务和功能。我们支持一组紧凑的语言任务评估（代码致谢 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)），并删除了启动时的注册逻辑（针对所有模型和任务）以减少开销。现在 `lmms-eval` 仅启动必要的任务/模型。详情请查看[发布说明](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3)。
- [2024-08] 🎉🎉 欢迎新模型 [LLaVA-OneVision](https://huggingface.co/papers/2408.03326)、[Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162)，以及新任务 [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench)、[LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117)、[MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158)。我们为 llava-onevision 模型提供了 SGlang Runtime API 的新功能，请参考[文档](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/getting-started/commands.md)以加速推理。
- [2024-07] 👨‍💻👨‍💻 `lmms-eval/v0.2.1` 已升级以支持更多模型，包括 [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)、[InternVL-2](https://github.com/OpenGVLab/InternVL)、[VILA](https://github.com/NVlabs/VILA)，以及更多评估任务，例如 [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136)、[MLVU](https://arxiv.org/abs/2406.04264)、[WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data)、[VITATECS](https://github.com/lscpku/VITATECS) 和 [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)。
- [2024-07] 🎉🎉 我们发布了[技术报告](https://arxiv.org/abs/2407.12772)和 [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)！
- [2024-06] 🎬🎬 `lmms-eval/v0.2.0` 已升级，支持对 LLaVA-NeXT Video 和 Gemini 1.5 Pro 等视频模型在 EgoSchema、PerceptionTest、VideoMME 等任务上进行视频评估。详情请参阅[博客](https://lmms-lab.github.io/posts/lmms-eval-0.2/)！
- [2024-03] 📝📝 我们发布了 `lmms-eval` 的第一个版本，详情请参阅[博客](https://lmms-lab.github.io/posts/lmms-eval-0.1/)！

</details>

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

**LLaVA-OneVision1_5 的评估**

```bash
bash examples/models/llava_onevision1_5.sh
```

**LLaMA-3.2-Vision 的评估**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL 的评估**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**针对大模型使用张量并行进行评估 (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**针对大模型使用 SGLang 进行评估 (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**针对大模型使用 vLLM 进行评估 (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**更多参数**

```bash
python3 -m lmms_eval --help
```

**环境变量**
在运行实验和评估之前，我们建议您向环境中导出以下环境变量。某些任务的运行需要其中的一些变量。

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# 其他可能的环境变量包括 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY 等。
```

**常见环境问题**

有时您可能会遇到一些常见问题，例如与 httpx 或 protobuf 相关的错误。要解决这些问题，您可以首先尝试：

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# 如果您使用的是 numpy==2.x，有时可能会导致错误
python3 -m pip install numpy==1.26;
# 有时需要 sentencepiece 才能使分词器工作
python3 -m pip install sentencepiece;
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
