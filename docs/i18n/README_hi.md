<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
</p>

# बड़े मल्टीमॉडल मॉडल मूल्यांकन सूट

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | **हिन्दी** | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> `lmms-eval` के साथ बड़े मल्टीमॉडल मॉडल (LMMs) के विकास को तेज करें। हम अधिकांश टेक्स्ट, इमेज, वीडियो और ऑडियो कार्यों का समर्थन करते हैं।

🏠 [LMMs-Lab होमपेज](https://www.lmms-lab.com/) | 🤗 [Huggingface डेटासेट](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [समर्थित कार्य (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [समर्थित मॉडल (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [दस्तावेज़ीकरण](../README.md)

---

## घोषणाएं

मल्टीमॉडल मॉडल का मूल्यांकन करना दिखने में जितना आसान है, उससे कहीं अधिक कठिन है। हमारे पास सैकड़ों बेंचमार्क हैं, लेकिन उन्हें चलाने का कोई मानक तरीका नहीं है। प्रयोगशालाओं के बीच परिणाम भिन्न होते हैं। तुलनाएं अविश्वसनीय हो जाती हैं। हम इस समस्या के समाधान के लिए काम कर रहे हैं - किसी वीरतापूर्ण प्रयास के माध्यम से नहीं, बल्कि व्यवस्थित प्रक्रिया के माध्यम से।

**जनवरी 2026** - हमने पहचाना कि स्थानिक (spatial) और रचनात्मक (compositional) तर्क मौजूदा बेंचमार्क में छिपे हुए क्षेत्र (blind spots) बने हुए थे। हमने [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/), और [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial) को जोड़ा है। रिमोट इवैल्यूएशन पाइपलाइन चलाने वाली टीमों के लिए, हमने एक HTTP इवैल्यूएशन सर्वर (#972) पेश किया है। सांख्यिकीय सटीकता की आवश्यकता वालों के लिए, हमने CLT और क्लस्टर्ड स्टैंडर्ड एरर अनुमान (#989) जोड़ा है।

**अक्टूबर 2025 (v0.5)** - ऑडियो एक खाली जगह थी। मॉडल सुन सकते थे, लेकिन हमारे पास उनका परीक्षण करने का कोई सुसंगत तरीका नहीं था। इस रिलीज़ ने व्यापक ऑडियो मूल्यांकन, दक्षता के लिए रिस्पॉन्स कैशिंग, और ऑडियो, विज़न और तर्क में फैले 50+ बेंचमार्क वेरिएंट जोड़े। [रिलीज़ नोट्स](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md)।

<details>
<summary>नीचे हमारे अद्भुत योगदानकर्ताओं द्वारा जोड़े गए हालिया कार्यों, मॉडलों और सुविधाओं की कालानुक्रमिक सूची दी गई है। </summary>

- [2025-01] 🎓🎓 हमने अपना नया बेंचमार्क जारी किया है: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826)। अधिक विवरण के लिए कृपया [प्रोजेक्ट पेज](https://videommmu.github.io/) देखें।
- [2024-12] 🎉🎉 हमने [MME टीम](https://github.com/BradyFU/Video-MME) और [OpenCompass टीम](https://github.com/open-compass) के साथ संयुक्त रूप से [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296) प्रस्तुत किया है।
- [2024-11] 🔈🔊 `lmms-eval/v0.3.0` को Qwen2-Audio और Gemini-Audio जैसे ऑडियो मॉडलों के लिए AIR-Bench, Clotho-AQA, LibriSpeech जैसे कार्यों में ऑडियो मूल्यांकन का समर्थन करने के लिए अपग्रेड किया गया है। अधिक विवरण के लिए कृपया [ब्लॉग](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.3.md) देखें!
- [2024-10] 🎉🎉 हम नए कार्य [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) का स्वागत करते हैं, जो एक विज़न-केंद्रित VQA बेंचमार्क (NeurIPS'24) है जो प्राकृतिक इमेजरी के बारे में सरल प्रश्नों के साथ विज़न-लैंग्वेज मॉडल को चुनौती देता है।
- [2024-10] 🎉🎉 हम वीडियो की सूक्ष्म समझ और तर्क के लिए नए कार्य [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) का स्वागत करते हैं, जो एक बड़ा (>30%) मानव-AI अंतर प्रकट करता है।
- [2024-10] 🎉🎉 हम वीडियो विस्तृत कैप्शनिंग के लिए नए कार्यों [VDC](https://rese1f.github.io/aurora-web/), लंबी अवधि की वीडियो समझ के लिए [MovieChat-1K](https://rese1f.github.io/MovieChat/), और [Vinoground](https://vinoground.github.io/), जो 1000 छोटे प्राकृतिक वीडियो-कैप्शन जोड़ों से बना एक टेंपोरल काउंटरफैक्चुअल LMM बेंचमार्क है, का स्वागत करते हैं। हम नए मॉडलों: [AuroraCap](https://github.com/rese1f/aurora) और [MovieChat](https://github.com/rese1f/MovieChat) का भी स्वागत करते हैं।
- [2024-09] 🎉🎉 हम अनुमान त्वरण के लिए नए कार्यों [MMSearch](https://mmsearch.github.io/) और [MME-RealWorld](https://mme-realworld.github.io/) का स्वागत करते हैं।
- [2024-09] ⚙️️⚙️️️️ हम `lmms-eval` को `0.2.3` में अधिक कार्यों और सुविधाओं के साथ अपग्रेड करते हैं। हम भाषा कार्यों के मूल्यांकन के एक संक्षिप्त सेट का समर्थन करते हैं (कोड क्रेडिट [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) को), और हम ओवरहेड को कम करने के लिए शुरुआत में पंजीकरण तर्क को हटा देते हैं। अब `lmms-eval` केवल आवश्यक कार्यों/मॉडलों को लॉन्च करता है। अधिक विवरण के लिए कृपया [रिलीज़ नोट्स](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) देखें।
- [2024-08] 🎉🎉 हम नए मॉडल [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), नए कार्यों [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158) का स्वागत करते हैं। हम llava-onevision मॉडल के लिए SGlang रनटाइम API की नई सुविधा प्रदान करते हैं, कृपया अनुमान त्वरण के लिए [दस्तावेज़](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/getting-started/commands.md) देखें।
- [2024-07] 👨‍💻👨‍💻 `lmms-eval/v0.2.1` को [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA) सहित अधिक मॉडलों और कई मूल्यांकन कार्यों, जैसे [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) और [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/) का समर्थन करने के लिए अपग्रेड किया गया है।
- [2024-07] 🎉🎉 हमने [तकनीकी रिपोर्ट](https://arxiv.org/abs/2407.12772) और [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench) जारी किया है!
- [2024-06] 🎬🎬 `lmms-eval/v0.2.0` को LLaVA-NeXT Video और Gemini 1.5 Pro जैसे वीडियो मॉडलों के लिए EgoSchema, PerceptionTest, VideoMME जैसे कार्यों में वीडियो मूल्यांकन का समर्थन करने के लिए अपग्रेड किया गया है। अधिक विवरण के लिए कृपया [ब्लॉग](https://lmms-lab.github.io/posts/lmms-eval-0.2/) देखें!
- [2024-03] 📝📝 हमने `lmms-eval` का पहला संस्करण जारी किया है, अधिक विवरण के लिए कृपया [ब्लॉग](https://lmms-lab.github.io/posts/lmms-eval-0.1/) देखें!

</details>

## `lmms-eval` क्यों?

हम कुछ ऐसा करने के बीच में हैं जो 1960 के दशक की अंतरिक्ष दौड़ जैसा महसूस होता है - सिवाय इसके कि इस बार, गंतव्य कृत्रिम सामान्य बुद्धिमत्ता (artificial general intelligence) है। बड़े मल्टीमॉडल मॉडल हमारे रॉकेट हैं। वे देख सकते हैं, सुन सकते हैं, पढ़ सकते हैं और तर्क कर सकते हैं। प्रगति वास्तविक है और तेज हो रही है।

लेकिन यहाँ समस्या यह है: हमारी माप प्रणाली हमारी महत्वाकांक्षाओं के साथ तालमेल नहीं रख पाई है।

हमारे पास बेंचमार्क हैं - उनमें से सैकड़ों। लेकिन वे Google Drive फ़ोल्डर्स, Dropbox लिंक, विश्वविद्यालय की वेबसाइटों और लैब सर्वरों में बिखरे हुए हैं। प्रत्येक बेंचमार्क का अपना डेटा प्रारूप, अपनी मूल्यांकन स्क्रिप्ट, अपनी विचित्रताएँ होती हैं। जब दो टीमें एक ही बेंचमार्क पर परिणाम रिपोर्ट करती हैं, तो उन्हें अक्सर अलग-अलग नंबर मिलते हैं। इसलिए नहीं कि उनके मॉडल अलग हैं, बल्कि इसलिए कि उनके मूल्यांकन पाइपलाइन अलग हैं।

कल्पना कीजिए कि यदि अंतरिक्ष दौड़ के दौरान, प्रत्येक देश अलग-अलग इकाइयों में दूरी मापता और अपनी रूपांतरण तालिकाओं को कभी साझा नहीं करता। मल्टीमॉडल मूल्यांकन के साथ आज हम मोटे तौर पर वहीं हैं।

यह कोई मामूली असुविधा नहीं है। यह एक प्रणालीगत विफलता है। निरंतर माप के बिना, हम यह नहीं जान सकते कि कौन से मॉडल वास्तव में बेहतर हैं। हम परिणामों को पुनरुत्पादित (reproduce) नहीं कर सकते। हम एक-दूसरे के काम पर निर्माण नहीं कर सकते।

भाषा मॉडलों के लिए, यह समस्या काफी हद तक [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) द्वारा हल की गई थी। यह एकीकृत डेटा लोडिंग, मानकीकृत मूल्यांकन और पुनरुत्पादनीय परिणाम प्रदान करता है। यह [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) को शक्ति प्रदान करता है। यह बुनियादी ढांचा बन गया है।

हमने मल्टीमॉडल मॉडलों के लिए भी यही करने के लिए `lmms-eval` बनाया है। वही सिद्धांत: एक फ्रेमवर्क, सुसंगत इंटरफेस, पुनरुत्पादनीय संख्याएं। मूनशॉट को एक विश्वसनीय पैमाने की आवश्यकता है।

## स्थापना

### uv का उपयोग करना (सुसंगत वातावरण के लिए अनुशंसित)

हम पैकेज प्रबंधन के लिए `uv` का उपयोग करते हैं ताकि यह सुनिश्चित हो सके कि सभी डेवलपर्स बिल्कुल समान पैकेज संस्करणों का उपयोग करें। पहले, uv स्थापित करें:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

सुसंगत वातावरण के साथ विकास के लिए:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# अनुशंसित
uv pip install -e ".[all]"
# यदि आप uv sync का उपयोग करना चाहते हैं
# uv sync  # यह uv.lock से आपके वातावरण को बनाता/अपडेट करता है
```

कमांड चलाने के लिए:
```bash
uv run python -m lmms_eval --help  # uv run के साथ कोई भी कमांड चलाएं
```

### वैकल्पिक स्थापना

Git से सीधे उपयोग के लिए:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# यदि आप इस स्थापना का उपयोग करते हैं तो आपको अपना खुद का टास्क yaml जोड़ना और शामिल करना पड़ सकता है
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## उपयोग

> अधिक उदाहरण [examples/models](../../examples/models) में देखें

**OpenAI संगत मॉडल का मूल्यांकन**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM का मूल्यांकन**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision का मूल्यांकन**

```bash
bash examples/models/llava_onevision.sh
```

**LLaVA-OneVision1_5 का मूल्यांकन**

```bash
bash examples/models/llava_onevision1_5.sh
```

**LLaMA-3.2-Vision का मूल्यांकन**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2.5-VL का मूल्यांकन**

```bash
bash examples/models/qwen2_5_vl.sh
```

**बड़े मॉडल के लिए टेंसर पैरेलल (tensor parallel) के साथ मूल्यांकन (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**बड़े मॉडल के लिए SGLang के साथ मूल्यांकन (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**अधिक पैरामीटर**

```bash
python3 -m lmms_eval --help
```

**पर्यावरण चर (Environmental Variables)**

प्रयोगों और मूल्यांकनों को चलाने से पहले, हम आपको अपने वातावरण में निम्नलिखित पर्यावरण चर (environment variables) निर्यात करने की सलाह देते हैं। कुछ कार्यों को चलाने के लिए इनमें से कुछ आवश्यक हैं।

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# अन्य संभावित पर्यावरण चरों में शामिल हैं 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY आदि।
```

**सामान्य वातावरण संबंधी समस्याएँ (Common Environment Issues)**

कभी-कभी आपको कुछ सामान्य समस्याओं का सामना करना पड़ सकता है, उदाहरण के लिए httpx या protobuf से संबंधित त्रुटि। इन समस्याओं को हल करने के लिए, आप पहले यह प्रयास कर सकते हैं:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# यदि आप numpy==2.x का उपयोग कर रहे हैं, तो कभी-कभी त्रुटियां हो सकती हैं
python3 -m pip install numpy==1.26;
# कभी-कभी टोकेनाइज़र के काम करने के लिए sentencepiece की आवश्यकता होती है
python3 -m pip install sentencepiece;
```

## कस्टम मॉडल और डेटासेट जोड़ें

हमारा [दस्तावेज़ीकरण](../README.md) देखें।

## आभार

lmms_eval [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) का एक फोर्क है। हम प्रासंगिक जानकारी के लिए lm-eval-harness के [दस्तावेज़ीकरण](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) को पढ़ने की सलाह देते हैं।

## उद्धरण

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
