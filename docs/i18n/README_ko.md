<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# 대규모 멀티모달 모델 평가 스위트

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | **한국어** | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> `lmms-eval`로 대규모 멀티모달 모델(LMMs) 개발을 가속화하세요. 텍스트, 이미지, 비디오, 오디오 태스크를 지원합니다.

🏠 [LMMs-Lab 홈페이지](https://www.lmms-lab.com/) | 🤗 [Huggingface 데이터셋](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [지원 태스크 (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [지원 모델 (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [문서](../README.md)

---

## 공지

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** 출시! 이 메이저 릴리스에서는 포괄적인 오디오 평가, 응답 캐싱, 5개의 새 모델(GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme) 및 오디오(Step2, VoiceBench, WenetSpeech), 비전(CharXiv, Lemonade), 추론(CSBench, SciBench, MedQA, SuperGPQA)에 걸친 50개 이상의 새로운 벤치마크 변형을 도입합니다. 자세한 내용은 [릴리스 노트](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md)를 참조하세요.
- [2025-07] 🚀🚀 `lmms-eval-0.4`를 출시했습니다. 자세한 내용은 [릴리스 노트](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)를 참조하세요.

## 왜 `lmms-eval`인가?

우리는 1960년대 달 착륙의 열정처럼 인공일반지능(AGI) 창조를 향한 흥미진진한 여정을 걷고 있습니다. 이 여정은 다양한 인간 작업을 이해하고, 배우고, 수행할 수 있는 복잡한 시스템인 고급 대규모 언어 모델(LLMs)과 대규모 멀티모달 모델(LMMs)에 의해 추진됩니다.

이러한 모델이 얼마나 발전했는지 측정하기 위해 다양한 평가 벤치마크를 사용합니다. 이러한 벤치마크는 이러한 모델의 기능을 이해하고 AGI 달성에 얼마나 가까운지 보여주는 도구입니다. 그러나 이러한 벤치마크를 찾고 사용하는 것은 큰 도전입니다.

언어 모델 분야에서는 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)의 선례가 귀중한 이정표가 되었습니다. 우리는 lm-evaluation-harness의 정교하고 효율적인 설계를 흡수하여 LMM의 일관되고 효율적인 평가를 위해 세심하게 만들어진 평가 프레임워크인 **lmms-eval**을 도입했습니다.

## 설치

### uv 사용 (일관된 환경에 권장)

모든 개발자가 정확히 동일한 패키지 버전을 사용할 수 있도록 `uv`를 패키지 관리에 사용합니다. 먼저 uv를 설치하세요:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

일관된 환경으로 개발:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# 권장
uv pip install -e ".[all]"
# uv sync를 사용하려면
# uv sync  # uv.lock에서 환경을 생성/업데이트합니다
```

명령 실행:
```bash
uv run python -m lmms_eval --help  # uv run으로 모든 명령 실행
```

### 대체 설치 방법

Git에서 직접 사용:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# 이 설치 방법을 사용하는 경우 자체 태스크 yaml을 추가하고 포함해야 할 수 있습니다
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## 사용법

> 더 많은 예제는 [examples/models](../../examples/models)를 참조하세요

**OpenAI 호환 모델 평가**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM 평가**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision 평가**

```bash
bash examples/models/llava_onevision.sh
```

**추가 파라미터**

```bash
python3 -m lmms_eval --help
```

## 사용자 정의 모델 및 데이터셋 추가

[문서](../README.md)를 참조하세요.

## 감사의 말

lmms_eval은 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)의 포크입니다. 관련 정보는 lm-eval-harness의 [문서](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)를 읽어보시기 바랍니다.

## 인용

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
