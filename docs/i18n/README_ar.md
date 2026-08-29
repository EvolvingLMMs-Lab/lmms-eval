<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
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

📖 [المهام المدعومة (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [النماذج المدعومة (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [التوثيق](../README.md)

---

## ما الجديد

تقييم النماذج متعددة الوسائط أصعب مما يبدو. لدينا مئات المعايير، ولكن لا توجد طريقة قياسية لتشغيلها. تختلف النتائج بين المختبرات، وتصبح المقارنات غير موثوقة. لقد عملنا على معالجة هذا الأمر - ليس من خلال جهد بطولي، بل من خلال عملية منهجية.

**يناير 2026** - أدركنا أن الاستدلال المكاني والتركيبي لا يزال يمثل نقاطًا عمياء في المعايير الحالية. أضفنا [CaptionQA](https://captionqa.github.io/) و [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench) و [SiteBench](https://sitebench.github.io/) و [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). بالنسبة للفرق التي تدير خطوط أنابيب تقييم عن بعد، قدمنا خادم تقييم HTTP (#972). ولمن يحتاجون إلى دقة إحصائية، أضفنا CLT وتقدير الخطأ المعياري العنقودي (#989).

**أكتوبر 2025 (v0.5)** - كان الصوت يمثل ثغرة. كان بإمكان النماذج السماع، ولكن لم تكن لدينا طريقة متسقة لاختبارها. أضاف هذا الإصدار تقييمًا صوتيًا شاملاً، وتخزينًا مؤقتًا للاستجابة من أجل الكفاءة، وأكثر من 50 متغيرًا للمعايير تغطي الصوت والرؤية والاستدلال. [ملاحظات الإصدار](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md).

<details>
<summary>فيما يلي قائمة زمنية بالمهام والنماذج والميزات الأخيرة التي أضافها مساهمونا الرائعون. </summary>

- [2025-07] 🚀🚀 أصدرنا `lmms-eval-0.4`. راجع [ملاحظات الإصدار](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.4.md) لمزيد من التفاصيل.

</details>

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

**تقييم LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**تقييم LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**تقييم Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**التقييم مع التوازي التوتري (tensor parallel) للنماذج الأكبر (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**التقييم مع SGLang للنماذج الأكبر (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**المزيد من المعلمات**

```bash
python3 -m lmms_eval --help
```

**متغيرات البيئة**
قبل تشغيل التجارب والتقييمات، نوصيك بتصدير متغيرات البيئة التالية إلى بيئتك. بعضها ضروري لتشغيل مهام معينة.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```

**مشاكل البيئة الشائعة**

أحيانًا قد تواجه بعض المشاكل الشائعة، على سبيل المثال خطأ متعلق بـ httpx أو protobuf. لحل هذه المشاكل، يمكنك أولاً تجربة

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# إذا كنت تستخدم numpy==2.x، فقد يتسبب ذلك أحيانًا في حدوث أخطاء
python3 -m pip install numpy==1.26;
# أحيانًا تكون sentencepiece مطلوبة ليعمل المحلل اللغوي (tokenizer)
python3 -m pip install sentencepiece;
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
