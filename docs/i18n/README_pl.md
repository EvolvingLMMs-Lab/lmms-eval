<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
</p>

# Pakiet Ewaluacyjny dla Dużych Modeli Multimodalnych

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | **Polski** | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Przyspieszenie rozwoju dużych modeli multimodalnych (LMMs) z `lmms-eval`. Obsługujemy większość zadań tekstowych, obrazowych, wideo i audio.

🏠 [Strona Główna LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Zbiory Danych Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Obsługiwane Zadania (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [Obsługiwane Modele (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Dokumentacja](../README.md)

---

## Ogłoszenia

**Styczeń 2026** - Rozpoznaliśmy, że rozumowanie przestrzenne i kompozycyjne pozostało martwym punktem w istniejących benchmarkach. Dodaliśmy [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/) oraz [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Dla zespołów prowadzących zdalne procesy ewaluacji wprowadziliśmy serwer HTTP eval (#972). Dla tych, którzy potrzebują rygoru statystycznego, dodaliśmy CLT i estymację zgrupowanego błędu standardowego (#989).

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** jest tutaj! Ta główna wersja wprowadza kompleksową ewaluację audio, buforowanie odpowiedzi, 5 nowych modeli (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme) oraz ponad 50 nowych wariantów benchmarków obejmujących audio (Step2, VoiceBench, WenetSpeech), wizję (CharXiv, Lemonade) i rozumowanie (CSBench, SciBench, MedQA, SuperGPQA). Szczegóły w [notatkach wydania](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md).
- [2025-07] 🚀🚀 Wydaliśmy `lmms-eval-0.4`. Szczegóły w [notatkach wydania](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.4.md).

## Dlaczego `lmms-eval`?

Jesteśmy w ekscytującej podróży ku stworzeniu Sztucznej Ogólnej Inteligencji (AGI), podobnej do entuzjazmu lądowania na Księżycu w latach 60. Ta podróż jest napędzana przez zaawansowane duże modele językowe (LLMs) i duże modele multimodalne (LMMs), złożone systemy zdolne do rozumienia, uczenia się i wykonywania szerokiej gamy ludzkich zadań.

Aby zmierzyć, jak zaawansowane są te modele, używamy różnych benchmarków ewaluacyjnych. Te benchmarki są narzędziami, które pomagają nam zrozumieć możliwości tych modeli, pokazując, jak blisko jesteśmy osiągnięcia AGI. Jednak znalezienie i wykorzystanie tych benchmarków jest dużym wyzwaniem.

W dziedzinie modeli językowych praca [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ustanowiła cenny precedens. Przyswoiliśmy wyrafinowany i efektywny design lm-evaluation-harness i wprowadziliśmy **lmms-eval**, starannie opracowany framework ewaluacyjny do spójnej i efektywnej ewaluacji LMM.

## Instalacja

### Używając uv (Zalecane dla spójnych środowisk)

Używamy `uv` do zarządzania pakietami, aby zapewnić, że wszyscy programiści używają dokładnie tych samych wersji pakietów. Najpierw zainstaluj uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Do rozwoju ze spójnym środowiskiem:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Zalecane
uv pip install -e ".[all]"
# Jeśli chcesz używać uv sync
# uv sync  # To tworzy/aktualizuje twoje środowisko z uv.lock
```

Aby uruchamiać polecenia:
```bash
uv run python -m lmms_eval --help  # Uruchom dowolne polecenie z uv run
```

### Alternatywna Instalacja

Do bezpośredniego użycia z Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Możesz potrzebować dodać i dołączyć własny yaml zadań, jeśli używasz tej instalacji
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Użycie

> Więcej przykładów w [examples/models](../../examples/models)

**Ewaluacja Modelu Kompatybilnego z OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Ewaluacja vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Ewaluacja LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Ewaluacja LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Ewaluacja LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Ewaluacja Qwen2.5-VL**

```bash
bash examples/models/qwen2_5_vl.sh
```

**Ewaluacja z tensor parallel dla większego modelu (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Ewaluacja z SGLang dla większego modelu (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Więcej Parametrów**

```bash
python3 -m lmms_eval --help
```

**Zmienne Środowiskowe**
Przed uruchomieniem eksperymentów i ewaluacji, zalecamy wyeksportowanie następujących zmiennych środowiskowych do środowiska. Niektóre są niezbędne do działania określonych zadań.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Inne możliwe zmienne środowiskowe to m.in. 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY itp.
```

**Typowe Problemy ze Środowiskiem**

Czasami możesz napotkać typowe problemy, na przykład błędy związane z httpx lub protobuf. Aby rozwiązać te problemy, możesz najpierw spróbować:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Jeśli używasz numpy==2.x, może to czasami powodować błędy
python3 -m pip install numpy==1.26;
# Czasami sentencepiece jest wymagane do działania tokenizera
python3 -m pip install sentencepiece;
```

## Dodawanie Niestandardowego Modelu i Zbioru Danych

Zobacz naszą [dokumentację](../README.md).

## Podziękowania

lmms_eval jest forkiem [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Zalecamy przeczytanie [dokumentacji lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) w celu uzyskania istotnych informacji.

## Cytowania

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
