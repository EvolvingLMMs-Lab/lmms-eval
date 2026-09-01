<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
</p>

# Набор Инструментов для Оценки Больших Мультимодальных Моделей

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | **Русский** | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Ускорение разработки больших мультимодальных моделей (LMMs) с помощью `lmms-eval`. Мы поддерживаем большинство задач с текстом, изображениями, видео и аудио.

🏠 [Главная страница LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Наборы данных Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Поддерживаемые задачи (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [Поддерживаемые модели (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Документация](../README.md)

---

## Объявления

**Январь 2026** — Мы признали, что пространственное и композиционное рассуждение оставались слепыми зонами в существующих бенчмарках. Мы добавили [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/) и [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Для команд, использующих удаленные конвейеры оценки, мы представили HTTP-сервер для оценки (#972). Для тех, кому нужна статистическая строгость, мы добавили CLT и оценку кластеризованной стандартной ошибки (#989).

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** здесь! Этот крупный релиз включает комплексную оценку аудио, кэширование ответов, 5 новых моделей (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme) и более 50 новых вариантов бенчмарков, охватывающих аудио (Step2, VoiceBench, WenetSpeech), зрение (CharXiv, Lemonade) и рассуждения (CSBench, SciBench, MedQA, SuperGPQA). Подробности см. в [примечаниях к релизу](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md).
- [2025-07] 🚀🚀 Мы выпустили `lmms-eval-0.4`. Подробности см. в [примечаниях к релизу](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.4.md).

## Почему `lmms-eval`?

Мы находимся на захватывающем пути к созданию Искусственного Общего Интеллекта (AGI), подобно энтузиазму высадки на Луну 1960-х годов. Этот путь движим продвинутыми большими языковыми моделями (LLMs) и большими мультимодальными моделями (LMMs), сложными системами, способными понимать, учиться и выполнять широкий спектр человеческих задач.

Для измерения того, насколько продвинуты эти модели, мы используем различные бенчмарки оценки. Эти бенчмарки — инструменты, помогающие нам понять возможности этих моделей, показывая, насколько мы близки к достижению AGI. Однако поиск и использование этих бенчмарков представляет большую проблему.

В области языковых моделей работа [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) создала ценный прецедент. Мы усвоили изысканный и эффективный дизайн lm-evaluation-harness и представили **lmms-eval**, тщательно разработанный фреймворк оценки для согласованной и эффективной оценки LMM.

## Установка

### Использование uv (Рекомендуется для согласованных окружений)

Мы используем `uv` для управления пакетами, чтобы гарантировать, что все разработчики используют точно такие же версии пакетов. Сначала установите uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Для разработки с согласованным окружением:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Рекомендуется
uv pip install -e ".[all]"
# Если вы хотите использовать uv sync
# uv sync  # Это создает/обновляет ваше окружение из uv.lock
```

Для запуска команд:
```bash
uv run python -m lmms_eval --help  # Запустить любую команду с uv run
```

### Альтернативная установка

Для прямого использования из Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Возможно, вам потребуется добавить и включить собственный yaml задач при использовании этой установки
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Использование

> Больше примеров в [examples/models](../../examples/models)

**Оценка модели, совместимой с OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Оценка vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Оценка LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Оценка LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Оценка LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Оценка Qwen2.5-VL**

```bash
bash examples/models/qwen2_5_vl.sh
```

**Оценка с использованием тензорного параллелизма для больших моделей (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Оценка с использованием SGLang для больших моделей (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Дополнительные параметры**

```bash
python3 -m lmms_eval --help
```

**Переменные окружения**
Перед запуском экспериментов и оценок мы рекомендуем экспортировать следующие переменные окружения. Некоторые из них необходимы для работы определенных задач.

```bash
export OPENAI_API_KEY="<ВАШ_API_KEY>"
export HF_HOME="<Путь к кэшу HF>" 
export HF_TOKEN="<ВАШ_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<ВАШ_API_KEY>"
# Другие возможные переменные окружения включают 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY и т. д.
```

**Общие проблемы с окружением**

Иногда вы можете столкнуться с общими проблемами, например, ошибками, связанными с httpx или protobuf. Для решения этих проблем вы можете сначала попробовать:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Если вы используете numpy==2.x, это иногда может вызывать ошибки
python3 -m pip install numpy==1.26;
# Иногда для работы токенизатора требуется sentencepiece
python3 -m pip install sentencepiece;
```

## Добавление пользовательской модели и набора данных

См. нашу [документацию](../README.md).

## Благодарности

lmms_eval — это форк [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Рекомендуем прочитать [документацию lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) для получения соответствующей информации.

## Цитирование

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
