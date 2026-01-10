<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Suite de Evaluación de Modelos Multimodales de Gran Escala

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | **Español** | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Acelerando el desarrollo de modelos multimodales de gran escala (LMMs) con `lmms-eval`. Soportamos la mayoría de tareas de texto, imagen, video y audio.

🏠 [Página Principal de LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Conjuntos de Datos de Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Tareas Soportadas (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Modelos Soportados (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentación](../README.md)

---

## Anuncios

- [2025-10] 🚀🚀 ¡**LMMs-Eval v0.5** está aquí! Esta versión principal introduce evaluación de audio completa, caché de respuestas, 5 nuevos modelos (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), y más de 50 nuevas variantes de benchmark que abarcan audio (Step2, VoiceBench, WenetSpeech), visión (CharXiv, Lemonade) y razonamiento (CSBench, SciBench, MedQA, SuperGPQA). Consulte las [notas de la versión](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) para más detalles.
- [2025-07] 🚀🚀 Hemos lanzado `lmms-eval-0.4`. Consulte las [notas de la versión](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) para más detalles.

## ¿Por qué `lmms-eval`?

Estamos en un emocionante viaje hacia la creación de Inteligencia General Artificial (AGI), similar al entusiasmo del aterrizaje lunar de los años 60. Este viaje está impulsado por modelos de lenguaje de gran escala (LLMs) y modelos multimodales de gran escala (LMMs), sistemas complejos capaces de entender, aprender y realizar una amplia variedad de tareas humanas.

Para medir cuán avanzados son estos modelos, utilizamos una variedad de benchmarks de evaluación. Estos benchmarks son herramientas que nos ayudan a entender las capacidades de estos modelos, mostrándonos qué tan cerca estamos de lograr AGI. Sin embargo, encontrar y usar estos benchmarks es un gran desafío.

En el campo de los modelos de lenguaje, el trabajo de [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ha establecido un precedente valioso. Absorbimos el diseño exquisito y eficiente de lm-evaluation-harness e introducimos **lmms-eval**, un framework de evaluación meticulosamente elaborado para la evaluación consistente y eficiente de LMM.

## Instalación

### Usando uv (Recomendado para entornos consistentes)

Usamos `uv` para la gestión de paquetes para asegurar que todos los desarrolladores usen exactamente las mismas versiones de paquetes. Primero, instale uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para desarrollo con entorno consistente:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Recomendado
uv pip install -e ".[all]"
# Si desea usar uv sync
# uv sync  # Esto crea/actualiza su entorno desde uv.lock
```

Para ejecutar comandos:
```bash
uv run python -m lmms_eval --help  # Ejecutar cualquier comando con uv run
```

### Instalación Alternativa

Para uso directo desde Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Puede que necesite agregar e incluir su propio yaml de tareas si usa esta instalación
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Uso

> Más ejemplos en [examples/models](../../examples/models)

**Evaluación de Modelo Compatible con OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluación de vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluación de LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Más Parámetros**

```bash
python3 -m lmms_eval --help
```

## Agregar Modelo y Conjunto de Datos Personalizados

Consulte nuestra [documentación](../README.md).

## Reconocimientos

lmms_eval es un fork de [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Recomendamos leer la [documentación de lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) para información relevante.

## Citas

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
