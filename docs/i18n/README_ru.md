<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
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

📖 [Поддерживаемые задачи (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Поддерживаемые модели (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Документация](../README.md)

---

## Объявления

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** здесь! Этот крупный релиз включает комплексную оценку аудио, кэширование ответов, 5 новых моделей (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme) и более 50 новых вариантов бенчмарков, охватывающих аудио (Step2, VoiceBench, WenetSpeech), зрение (CharXiv, Lemonade) и рассуждения (CSBench, SciBench, MedQA, SuperGPQA). Подробности см. в [примечаниях к релизу](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md).
- [2025-07] 🚀🚀 Мы выпустили `lmms-eval-0.4`. Подробности см. в [примечаниях к релизу](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).

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

**Дополнительные параметры**

```bash
python3 -m lmms_eval --help
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
```
