<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Suite Evaluasi Model Multimodal Besar

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | **Indonesia**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Mempercepat pengembangan model multimodal besar (LMMs) dengan `lmms-eval`. Kami mendukung sebagian besar tugas teks, gambar, video, dan audio.

🏠 [Beranda LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Dataset Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Tugas yang Didukung (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Model yang Didukung (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Dokumentasi](../README.md)

---

## Pengumuman

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** hadir! Rilis utama ini memperkenalkan evaluasi audio komprehensif, caching respons, 5 model baru (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), dan 50+ varian benchmark baru yang mencakup audio (Step2, VoiceBench, WenetSpeech), visi (CharXiv, Lemonade), dan penalaran (CSBench, SciBench, MedQA, SuperGPQA). Lihat [catatan rilis](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) untuk detail.
- [2025-07] 🚀🚀 Kami telah merilis `lmms-eval-0.4`. Lihat [catatan rilis](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) untuk detail lebih lanjut.

## Mengapa `lmms-eval`?

Kita sedang dalam perjalanan yang menarik menuju penciptaan Kecerdasan Buatan Umum (AGI), mirip dengan antusiasme pendaratan di bulan tahun 1960-an. Perjalanan ini didorong oleh model bahasa besar yang canggih (LLMs) dan model multimodal besar (LMMs), sistem kompleks yang mampu memahami, belajar, dan melakukan berbagai tugas manusia.

Untuk mengukur seberapa maju model-model ini, kami menggunakan berbagai benchmark evaluasi. Benchmark ini adalah alat yang membantu kami memahami kemampuan model-model ini, menunjukkan seberapa dekat kita dengan mencapai AGI. Namun, menemukan dan menggunakan benchmark ini adalah tantangan besar.

Di bidang model bahasa, karya [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) telah menetapkan preseden yang berharga. Kami menyerap desain yang indah dan efisien dari lm-evaluation-harness dan memperkenalkan **lmms-eval**, kerangka kerja evaluasi yang dibuat dengan cermat untuk evaluasi LMM yang konsisten dan efisien.

## Instalasi

### Menggunakan uv (Direkomendasikan untuk lingkungan yang konsisten)

Kami menggunakan `uv` untuk manajemen paket untuk memastikan semua pengembang menggunakan versi paket yang sama persis. Pertama, instal uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Untuk pengembangan dengan lingkungan yang konsisten:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Direkomendasikan
uv pip install -e ".[all]"
# Jika Anda ingin menggunakan uv sync
# uv sync  # Ini membuat/memperbarui lingkungan Anda dari uv.lock
```

Untuk menjalankan perintah:
```bash
uv run python -m lmms_eval --help  # Jalankan perintah apa pun dengan uv run
```

### Instalasi Alternatif

Untuk penggunaan langsung dari Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Anda mungkin perlu menambahkan dan menyertakan yaml tugas Anda sendiri jika menggunakan instalasi ini
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Penggunaan

> Lebih banyak contoh di [examples/models](../../examples/models)

**Evaluasi Model yang Kompatibel dengan OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluasi vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluasi LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Parameter Lainnya**

```bash
python3 -m lmms_eval --help
```

## Menambahkan Model dan Dataset Kustom

Lihat [dokumentasi](../README.md) kami.

## Ucapan Terima Kasih

lmms_eval adalah fork dari [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Kami menyarankan untuk membaca [dokumentasi lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) untuk informasi yang relevan.

## Sitasi

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
