<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
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

📖 [지원 태스크 (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [지원 모델 (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [문서](../README.md)

---

## 최신 소식

멀티모달 모델을 평가하는 것은 보기보다 어렵습니다. 수백 개의 벤치마크가 있지만 이를 실행하는 표준화된 방법은 없었습니다. 실험실마다 결과가 다르고 비교의 신뢰도가 떨어집니다. 우리는 영웅적인 노력이 아닌 체계적인 프로세스를 통해 이 문제를 해결하기 위해 노력해 왔습니다.

**2026년 1월** - 기존 벤치마크에서 공간 및 구성적 추론이 여전히 사각지대로 남아있음을 확인했습니다. 이에 [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/), [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial)을 추가했습니다. 원격 평가 파이프라인을 운영하는 팀들을 위해 HTTP 평가 서버(#972)를 도입했습니다. 통계적 엄밀함이 필요한 사용자들을 위해 CLT 및 클러스터링된 표준 오차 추정(#989)을 추가했습니다.

**2025년 10월 (v0.5)** - 오디오 분야는 그동안 공백이었습니다. 모델은 들을 수 있었지만 이를 테스트할 일관된 방법이 없었습니다. 이번 릴리스에서는 포괄적인 오디오 평가, 효율성을 위한 응답 캐싱, 그리고 오디오, 비전, 추론을 아우르는 50개 이상의 벤치마크 변형을 추가했습니다. [릴리스 노트](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md).

<details>
<summary>아래는 놀라운 기여자들에 의해 추가된 최근 태스크, 모델 및 기능의 연대순 목록입니다.</summary>

- [2025-07] 🚀🚀 `lmms-eval-0.4`를 출시했습니다. 자세한 내용은 [릴리스 노트](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.4.md)를 참조하세요.

</details>

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

**LLaVA-OneVision1_5 평가**

```bash
bash examples/models/llava_onevision1_5.sh
```

**LLaMA-3.2-Vision 평가**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL 평가**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**더 큰 모델을 위한 텐서 병렬(tensor parallel) 평가 (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**더 큰 모델을 위한 SGLang 평가 (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**더 큰 모델을 위한 vLLM 평가 (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**추가 파라미터**

```bash
python3 -m lmms_eval --help
```

**환경 변수**
실험 및 평가를 실행하기 전에 다음 환경 변수를 설정하는 것을 권장합니다. 일부 변수는 특정 태스크 실행에 필수적입니다.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# 기타 가능한 환경 변수는 다음과 같습니다:
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY 등
```

**일반적인 환경 문제**

가끔 httpx 또는 protobuf와 관련된 오류와 같은 일반적인 문제에 직면할 수 있습니다. 이러한 문제를 해결하기 위해 다음 명령어를 먼저 시도해 볼 수 있습니다:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# numpy==2.x를 사용하는 경우 오류가 발생할 수 있습니다
python3 -m pip install numpy==1.26;
# 토크나이저 작동을 위해 sentencepiece가 필요할 수 있습니다
python3 -m pip install sentencepiece;
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
