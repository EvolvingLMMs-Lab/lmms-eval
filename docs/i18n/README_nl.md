<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Evaluatiesuite voor Grote Multimodale Modellen

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | **Nederlands** | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Versnelling van de ontwikkeling van grote multimodale modellen (LMMs) met `lmms-eval`. We ondersteunen de meeste tekst-, beeld-, video- en audiotaken.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Ondersteunde Taken (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Ondersteunde Modellen (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentatie](../README.md)

---

## Aankondigingen

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** is hier! Deze belangrijke release introduceert uitgebreide audio-evaluatie, response caching, 5 nieuwe modellen (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), en meer dan 50 nieuwe benchmark-varianten die audio (Step2, VoiceBench, WenetSpeech), visie (CharXiv, Lemonade) en redeneren (CSBench, SciBench, MedQA, SuperGPQA) beslaan. Zie de [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) voor details.
- [2025-07] 🚀🚀 We hebben `lmms-eval-0.4` uitgebracht. Zie de [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) voor meer details.

## Waarom `lmms-eval`?

We zijn op een spannende reis naar het creëren van Kunstmatige Algemene Intelligentie (AGI), vergelijkbaar met het enthousiasme van de maanlanding in de jaren '60. Deze reis wordt aangedreven door geavanceerde grote taalmodellen (LLMs) en grote multimodale modellen (LMMs), complexe systemen die in staat zijn om een breed scala aan menselijke taken te begrijpen, te leren en uit te voeren.

Om te meten hoe geavanceerd deze modellen zijn, gebruiken we verschillende evaluatiebenchmarks. Deze benchmarks zijn hulpmiddelen die ons helpen de mogelijkheden van deze modellen te begrijpen, en ons laten zien hoe dicht we bij het bereiken van AGI zijn. Het vinden en gebruiken van deze benchmarks is echter een grote uitdaging.

Op het gebied van taalmodellen heeft het werk van [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) een waardevolle precedent gezet. We hebben het verfijnde en efficiënte ontwerp van lm-evaluation-harness geabsorbeerd en **lmms-eval** geïntroduceerd, een zorgvuldig ontworpen evaluatieframework voor consistente en efficiënte evaluatie van LMM.

## Installatie

### Met uv (Aanbevolen voor consistente omgevingen)

We gebruiken `uv` voor pakketbeheer om ervoor te zorgen dat alle ontwikkelaars exact dezelfde pakketversies gebruiken. Installeer eerst uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Voor ontwikkeling met consistente omgeving:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Aanbevolen
uv pip install -e ".[all]"
# Als je uv sync wilt gebruiken
# uv sync  # Dit maakt/update je omgeving vanuit uv.lock
```

Om commando's uit te voeren:
```bash
uv run python -m lmms_eval --help  # Voer elk commando uit met uv run
```

### Alternatieve Installatie

Voor direct gebruik vanuit Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Je moet mogelijk je eigen taak yaml toevoegen en opnemen als je deze installatie gebruikt
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Gebruik

> Meer voorbeelden in [examples/models](../../examples/models)

**Evaluatie van OpenAI-compatibel Model**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluatie van vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluatie van LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Meer Parameters**

```bash
python3 -m lmms_eval --help
```

## Aangepast Model en Dataset Toevoegen

Zie onze [documentatie](../README.md).

## Dankbetuigingen

lmms_eval is een fork van [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We raden aan om de [documentatie van lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) te lezen voor relevante informatie.

## Citaties

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
