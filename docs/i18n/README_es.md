<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
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

📖 [Tareas Soportadas (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [Modelos Soportados (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentación](../README.md)

---

## Anuncios

Evaluar modelos multimodales es más difícil de lo que parece. Tenemos cientos de benchmarks, pero no hay una forma estándar de ejecutarlos. Los resultados varían entre laboratorios. Las comparaciones se vuelven poco fiables. Hemos estado trabajando para abordar esto, no mediante esfuerzos heroicos, sino mediante procesos sistemáticos.

**Enero de 2026** - Reconocimos que el razonamiento espacial y composicional seguían siendo puntos ciegos en los benchmarks existentes. Añadimos [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/), y [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Para los equipos que ejecutan flujos de evaluación remotos, introdujimos un servidor de evaluación HTTP (#972). Para quienes necesitan rigor estadístico, añadimos CLT y estimación de error estándar por clúster (#989).

**Octubre de 2025 (v0.5)** - El audio había sido una brecha. Los modelos podían oír, pero no teníamos una forma consistente de probarlos. Esta versión añadió una evaluación de audio completa, caché de respuestas para mayor eficiencia y más de 50 variantes de benchmarks que abarcan audio, visión y razonamiento. [Notas de la versión](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md).

<details>
<summary>A continuación se presenta una lista cronológica de las tareas, modelos y características recientes añadidos por nuestros increíbles colaboradores.</summary>

- [2025-01] 🎓🎓 Hemos lanzado nuestro nuevo benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Consulte la [página del proyecto](https://videommmu.github.io/) para más detalles.
- [2024-12] 🎉🎉 Hemos presentado [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), conjuntamente con el [Equipo MME](https://github.com/BradyFU/Video-MME) y el [Equipo OpenCompass](https://github.com/open-compass).
- [2024-11] 🔈🔊 El `lmms-eval/v0.3.0` ha sido actualizado para soportar evaluaciones de audio para modelos de audio como Qwen2-Audio y Gemini-Audio en tareas como AIR-Bench, Clotho-AQA, LibriSpeech, y más. ¡Consulte el [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.3.md) para más detalles!

</details>

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

**Evaluación de LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Evaluación de LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluación de Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluación de LLaVA en MME**

Si desea probar LLaVA 1.5, tendrá que clonar su repositorio de [LLaVA](https://github.com/haotian-liu/LLaVA) y

```bash
bash examples/models/llava_next.sh
```

**Evaluación con tensor parallel para modelos más grandes (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluación con SGLang para modelos más grandes (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Más Parámetros**

```bash
python3 -m lmms_eval --help
```

## Variables de Entorno
Antes de ejecutar experimentos y evaluaciones, le recomendamos exportar las siguientes variables de entorno a su entorno. Algunas son necesarias para que ciertas tareas funcionen.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Otras posibles variables de entorno incluyen 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY etc.
```

## Problemas Comunes del Entorno

A veces puede encontrar algunos problemas comunes, por ejemplo, errores relacionados con httpx o protobuf. Para resolver estos problemas, primero puede intentar:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Si está usando numpy==2.x, a veces puede causar errores
python3 -m pip install numpy==1.26;
# A veces se requiere sentencepiece para que el tokenizador funcione
python3 -m pip install sentencepiece;
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
