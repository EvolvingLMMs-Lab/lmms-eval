<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

<div dir="rtl">

# مجموعة تقييم النماذج متعددة الوسائط الكبيرة

</div>

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | **العربية** | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> تسريع تطوير النماذج متعددة الوسائط الكبيرة (LMMs) باستخدام `lmms-eval`. نحن ندعم معظم مهام النص والصور والفيديو والصوت.

🏠 [الصفحة الرئيسية لـ LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [مجموعات بيانات Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [المهام المدعومة (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [النماذج المدعومة (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [التوثيق](../README.md)

---

## الإعلانات

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** متاح الآن! يقدم هذا الإصدار الرئيسي تقييم صوتي شامل، وتخزين مؤقت للاستجابات، و5 نماذج جديدة (GPT-4o Audio Preview، Gemma-3، LongViLA-R1، LLaVA-OneVision 1.5، Thyme)، وأكثر من 50 متغيرًا جديدًا للمعايير تغطي الصوت (Step2، VoiceBench، WenetSpeech)، والرؤية (CharXiv، Lemonade)، والاستدلال (CSBench، SciBench، MedQA، SuperGPQA). راجع [ملاحظات الإصدار](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) للتفاصيل.
- [2025-07] 🚀🚀 أصدرنا `lmms-eval-0.4`. راجع [ملاحظات الإصدار](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) لمزيد من التفاصيل.

## لماذا `lmms-eval`؟

نحن في رحلة مثيرة نحو إنشاء الذكاء الاصطناعي العام (AGI)، مشابهة لحماس الهبوط على القمر في الستينيات. هذه الرحلة مدعومة بنماذج اللغة الكبيرة المتقدمة (LLMs) والنماذج متعددة الوسائط الكبيرة (LMMs)، وهي أنظمة معقدة قادرة على فهم وتعلم وأداء مجموعة واسعة من المهام البشرية.

لقياس مدى تقدم هذه النماذج، نستخدم مجموعة متنوعة من معايير التقييم. هذه المعايير هي أدوات تساعدنا على فهم قدرات هذه النماذج، وتوضح لنا مدى قربنا من تحقيق AGI. ومع ذلك، فإن العثور على هذه المعايير واستخدامها يمثل تحديًا كبيرًا.

في مجال نماذج اللغة، وضع عمل [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) سابقة قيمة. لقد استوعبنا التصميم الرائع والفعال لـ lm-evaluation-harness وقدمنا **lmms-eval**، إطار عمل تقييم مصنوع بدقة لتقييم متسق وفعال لـ LMM.

## التثبيت

### باستخدام uv (موصى به للبيئات المتسقة)

نستخدم `uv` لإدارة الحزم لضمان استخدام جميع المطورين لنفس إصدارات الحزم بالضبط. أولاً، قم بتثبيت uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

للتطوير مع بيئة متسقة:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# موصى به
uv pip install -e ".[all]"
# إذا كنت تريد استخدام uv sync
# uv sync  # هذا ينشئ/يحدث بيئتك من uv.lock
```

لتشغيل الأوامر:
```bash
uv run python -m lmms_eval --help  # تشغيل أي أمر مع uv run
```

### التثبيت البديل

للاستخدام المباشر من Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# قد تحتاج إلى إضافة وتضمين yaml المهام الخاص بك إذا كنت تستخدم هذا التثبيت
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## الاستخدام

> المزيد من الأمثلة في [examples/models](../../examples/models)

**تقييم نموذج متوافق مع OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**تقييم vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**تقييم LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**المزيد من المعلمات**

```bash
python3 -m lmms_eval --help
```

## إضافة نموذج ومجموعة بيانات مخصصة

راجع [التوثيق](../README.md).

## شكر وتقدير

lmms_eval هو تفرع من [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). نوصي بقراءة [توثيق lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) للحصول على المعلومات ذات الصلة.

## الاستشهادات

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
