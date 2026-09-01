<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
</p>

# Büyük Çok Modlu Modeller için Değerlendirme Paketi

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | **Türkçe** | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> `lmms-eval` ile büyük çok modlu modellerin (LMMs) geliştirilmesini hızlandırın. Çoğu metin, görüntü, video ve ses görevini destekliyoruz.

🏠 [LMMs-Lab Ana Sayfa](https://www.lmms-lab.com/) | 🤗 [Huggingface Veri Setleri](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Desteklenen Görevler (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [Desteklenen Modeller (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Dokümantasyon](../README.md)

---

## Duyurular

- [2026-01] 🚀🚀 **Ocak 2026** - Mevcut kıyaslamalarda (benchmarks) uzamsal ve kompozisyonel akıl yürütmenin hala kör noktalar olduğunu fark ettik. [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/) ve [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial) benchmarklarını ekledik. Uzaktan değerlendirme boru hatları (pipeline) çalıştıran ekipler için bir HTTP değerlendirme sunucusu (#972) sunduk. İstatistiksel titizlik isteyenler için CLT ve kümelenmiş standart hata tahmini (#989) özelliklerini ekledik.
- [2025-10] 🚀🚀 **LMMs-Eval v0.5** burada! Bu büyük sürüm, kapsamlı ses değerlendirmesi, yanıt önbellekleme, 5 yeni model (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme) ve ses (Step2, VoiceBench, WenetSpeech), görüntü (CharXiv, Lemonade) ve akıl yürütme (CSBench, SciBench, MedQA, SuperGPQA) kapsayan 50'den fazla yeni benchmark varyantı sunuyor. Detaylar için [sürüm notlarına](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md) bakın.
- [2025-07] 🚀🚀 `lmms-eval-0.4` sürümünü yayınladık. Daha fazla detay için [sürüm notlarına](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.4.md) bakın.

## Neden `lmms-eval`?

1960'ların Ay'a iniş heyecanına benzer şekilde, Yapay Genel Zeka (AGI) yaratmaya doğru heyecan verici bir yolculuktayız. Bu yolculuk, çok çeşitli insan görevlerini anlama, öğrenme ve gerçekleştirme kapasitesine sahip karmaşık sistemler olan gelişmiş büyük dil modelleri (LLMs) ve büyük çok modlu modeller (LMMs) tarafından desteklenmektedir.

Bu modellerin ne kadar gelişmiş olduğunu ölçmek için çeşitli değerlendirme kıyaslamaları kullanıyoruz. Bu kıyaslamalar, bu modellerin yeteneklerini anlamamıza yardımcı olan, AGI'ye ne kadar yakın olduğumuzu gösteren araçlardır. Ancak, bu kıyaslamaları bulmak ve kullanmak büyük bir zorluktur.

Dil modelleri alanında, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) çalışması değerli bir emsal oluşturmuştur. lm-evaluation-harness'ın zarif ve verimli tasarımını benimsedik ve LMM'lerin tutarlı ve verimli değerlendirmesi için titizlikle hazırlanmış bir değerlendirme çerçevesi olan **lmms-eval**'i tanıttık.

## Kurulum

### uv Kullanarak (Tutarlı ortamlar için önerilir)

Tüm geliştiricilerin tam olarak aynı paket sürümlerini kullanmasını sağlamak için `uv` paket yöneticisini kullanıyoruz. Önce uv'yi kurun:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Tutarlı ortamla geliştirme için:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Önerilen
uv pip install -e ".[all]"
# uv sync kullanmak istiyorsanız
# uv sync  # Bu, uv.lock'tan ortamınızı oluşturur/günceller
```

Komutları çalıştırmak için:
```bash
uv run python -m lmms_eval --help  # Herhangi bir komutu uv run ile çalıştırın
```

### Alternatif Kurulum

Git'ten doğrudan kullanım için:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Bu kurulumu kullanıyorsanız kendi görev yaml'ınızı eklemeniz ve dahil etmeniz gerekebilir
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Kullanım

> Daha fazla örnek [examples/models](../../examples/models) içinde

**OpenAI Uyumlu Model Değerlendirmesi**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM Değerlendirmesi**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision Değerlendirmesi**

```bash
bash examples/models/llava_onevision.sh
```

**LLaVA-OneVision1_5 Değerlendirmesi**

```bash
bash examples/models/llava_onevision1_5.sh
```

**LLaMA-3.2-Vision Değerlendirmesi**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2.5-VL Değerlendirmesi**

```bash
bash examples/models/qwen2_5_vl.sh
```

**Büyük Model için Tensor Parallel Değerlendirmesi (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Büyük Model için SGLang Değerlendirmesi (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Daha Fazla Parametre**

```bash
python3 -m lmms_eval --help
```

**Ortam Değişkenleri**
Deneyleri ve değerlendirmeleri çalıştırmadan önce, aşağıdaki ortam değişkenlerini ortamınıza dışa aktarmanızı (export) öneririz. Bazıları belirli görevlerin çalışması için gereklidir.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```

**Yaygın Ortam Sorunları**

Bazen httpx veya protobuf ile ilgili hatalar gibi yaygın sorunlarla karşılaşabilirsiniz. Bu sorunları çözmek için önce şunları deneyebilirsiniz:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# numpy==2.x kullanıyorsanız bazen hatalara neden olabilir
python3 -m pip install numpy==1.26;
# Tokenizer'ın çalışması için bazen sentencepiece gereklidir
python3 -m pip install sentencepiece;
```

## Özel Model ve Veri Seti Ekleme

[Dokümantasyonumuza](../README.md) bakın.

## Teşekkürler

lmms_eval, [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)'in bir çatalıdır. İlgili bilgiler için lm-eval-harness [dokümantasyonunu](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) okumanızı öneririz.

## Atıflar

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

