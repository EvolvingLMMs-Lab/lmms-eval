<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
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

📖 [Unterstützte Aufgaben (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [Unterstützte Modelle (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Dokumentation](../README.md)

---

## Was ist neu?

Die Evaluierung multimodaler Modelle ist schwieriger, als es aussieht. Wir haben hunderte von Benchmarks, aber keinen Standardweg, um sie auszuführen. Die Ergebnisse variieren zwischen den Laboren. Vergleiche werden unzuverlässig. Wir haben daran gearbeitet, dies zu beheben – nicht durch heldenhaften Einsatz, sondern durch systematische Prozesse.

**Januar 2026** – Wir haben erkannt, dass räumliches und kompositionelles Denken blinde Flecken in bestehenden Benchmarks blieben. Wir haben [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/) und [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial) hinzugefügt. Für Teams, die Remote-Evaluierungs-Pipelines betreiben, haben wir einen HTTP-Eval-Server eingeführt (#972). Für diejenigen, die statistische Strenge benötigen, haben wir CLT und Clustered Standard Error Estimation hinzugefügt (#989).

**Oktober 2025 (v0.5)** – Audio war eine Lücke. Modelle konnten hören, aber wir hatten keinen konsistenten Weg, sie zu testen. Dieses Release fügte eine umfassende Audio-Evaluierung, Response-Caching für Effizienz und über 50 Benchmark-Varianten hinzu, die Audio, Vision und Reasoning abdecken. [Release Notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md).

<details>
<summary>Nachfolgend finden Sie eine chronologische Liste der jüngsten Aufgaben, Modelle und Funktionen, die von unseren großartigen Mitwirkenden hinzugefügt wurden. </summary>

- [2025-01] 🎓🎓 Wir haben unseren neuen Benchmark veröffentlicht: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Weitere Details finden Sie auf der [Projektseite](https://videommmu.github.io/).
- [2024-12] 🎉🎉 Wir haben gemeinsam mit dem [MME-Team](https://github.com/BradyFU/Video-MME) und dem [OpenCompass-Team](https://github.com/open-compass) den [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296) vorgestellt.
- [2024-11] 🔈🔊 `lmms-eval/v0.3.0` wurde aktualisiert, um Audio-Evaluierungen für Audio-Modelle wie Qwen2-Audio und Gemini-Audio über Aufgaben wie AIR-Bench, Clotho-AQA, LibriSpeech und mehr hinweg zu unterstützen. Weitere Details finden Sie im [Blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.3.md)!
- [2024-10] 🎉🎉 Wir begrüßen die neue Aufgabe [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), ein visionszentrierter VQA-Benchmark (NeurIPS'24), der Vision-Language-Modelle mit einfachen Fragen zu natürlichen Bildern herausfordert.
- [2024-10] 🎉🎉 Wir begrüßen die neue Aufgabe [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) für feingliedriges temporäres Verständnis und Schlussfolgern für Videos, die eine riesige (>30%) Lücke zwischen Mensch und KI aufdeckt.

</details>

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

**Evaluierung von LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Evaluierung von LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluierung von Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluierung von LLaVA auf MME**

Wenn Sie LLaVA 1.5 testen möchten, müssen Sie deren Repository von [LLaVA](https://github.com/haotian-liu/LLaVA) klonen und

```bash
bash examples/models/llava_next.sh
```

**Evaluierung mit Tensor Parallel für größere Modelle (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluierung mit SGLang für größere Modelle (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Evaluierung mit vLLM für größere Modelle (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Weitere Parameter**

```bash
python3 -m lmms_eval --help
```

**Umgebungsvariablen**
Bevor Sie Experimente und Evaluierungen durchführen, empfehlen wir Ihnen, die folgenden Umgebungsvariablen in Ihre Umgebung zu exportieren. Einige sind für die Ausführung bestimmter Aufgaben erforderlich.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Weitere mögliche Umgebungsvariablen sind 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY etc.
```

**Häufige Umgebungsprobleme**

Manchmal treten häufige Probleme auf, zum Beispiel Fehler im Zusammenhang mit `httpx` oder `protobuf`. Um diese Probleme zu lösen, können Sie zunächst versuchen:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Wenn Sie numpy==2.x verwenden, kann dies manchmal Fehler verursachen
python3 -m pip install numpy==1.26;
# Manchmal ist sentencepiece erforderlich, damit der Tokenizer funktioniert
python3 -m pip install sentencepiece;
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
