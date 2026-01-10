<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Evaluierungssuite für Große Multimodale Modelle

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | **Deutsch** | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Beschleunigung der Entwicklung großer multimodaler Modelle (LMMs) mit `lmms-eval`. Wir unterstützen die meisten Text-, Bild-, Video- und Audio-Aufgaben.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datensätze](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Unterstützte Aufgaben (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Unterstützte Modelle (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Dokumentation](../README.md)

---

## Ankündigungen

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** ist da! Diese Hauptversion führt umfassende Audio-Evaluierung, Response-Caching, 5 neue Modelle (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme) und über 50 neue Benchmark-Varianten ein, die Audio (Step2, VoiceBench, WenetSpeech), Vision (CharXiv, Lemonade) und Reasoning (CSBench, SciBench, MedQA, SuperGPQA) abdecken. Details finden Sie in den [Release Notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md).
- [2025-07] 🚀🚀 Wir haben `lmms-eval-0.4` veröffentlicht. Details finden Sie in den [Release Notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).

## Warum `lmms-eval`?

Wir befinden uns auf einer aufregenden Reise zur Schaffung Künstlicher Allgemeiner Intelligenz (AGI), ähnlich wie die Begeisterung der Mondlandung in den 1960er Jahren. Diese Reise wird von fortschrittlichen großen Sprachmodellen (LLMs) und großen multimodalen Modellen (LMMs) angetrieben, komplexen Systemen, die in der Lage sind, eine Vielzahl menschlicher Aufgaben zu verstehen, zu lernen und auszuführen.

Um zu messen, wie fortschrittlich diese Modelle sind, verwenden wir verschiedene Evaluierungs-Benchmarks. Diese Benchmarks sind Werkzeuge, die uns helfen, die Fähigkeiten dieser Modelle zu verstehen und zeigen, wie nah wir der Erreichung von AGI sind. Das Finden und Verwenden dieser Benchmarks ist jedoch eine große Herausforderung.

Im Bereich der Sprachmodelle hat die Arbeit von [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) einen wertvollen Präzedenzfall geschaffen. Wir haben das exquisite und effiziente Design von lm-evaluation-harness aufgenommen und **lmms-eval** eingeführt, ein sorgfältig entwickeltes Evaluierungs-Framework für konsistente und effiziente Evaluierung von LMM.

## Installation

### Verwendung von uv (Empfohlen für konsistente Umgebungen)

Wir verwenden `uv` für die Paketverwaltung, um sicherzustellen, dass alle Entwickler exakt dieselben Paketversionen verwenden. Installieren Sie zunächst uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Für die Entwicklung mit konsistenter Umgebung:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Empfohlen
uv pip install -e ".[all]"
# Wenn Sie uv sync verwenden möchten
# uv sync  # Dies erstellt/aktualisiert Ihre Umgebung aus uv.lock
```

Um Befehle auszuführen:
```bash
uv run python -m lmms_eval --help  # Beliebigen Befehl mit uv run ausführen
```

### Alternative Installation

Für direkte Verwendung von Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Möglicherweise müssen Sie Ihre eigene Task-YAML hinzufügen und einbinden, wenn Sie diese Installation verwenden
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Verwendung

> Weitere Beispiele in [examples/models](../../examples/models)

**Evaluierung eines OpenAI-kompatiblen Modells**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluierung von vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluierung von LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Weitere Parameter**

```bash
python3 -m lmms_eval --help
```

## Benutzerdefiniertes Modell und Datensatz Hinzufügen

Siehe unsere [Dokumentation](../README.md).

## Danksagungen

lmms_eval ist ein Fork von [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Wir empfehlen, die [Dokumentation von lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) für relevante Informationen zu lesen.

## Zitierung

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
