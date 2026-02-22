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

## Apa yang Baru

Mengevaluasi model multimodal lebih sulit daripada yang terlihat. Kami memiliki ratusan benchmark, tetapi tidak ada cara standar untuk menjalankannya. Hasil bervariasi antar lab. Perbandingan menjadi tidak dapat diandalkan. Kami telah bekerja untuk mengatasi hal ini - bukan melalui upaya heroik, tetapi melalui proses yang sistematis.

**Januari 2026** - Kami menyadari bahwa penalaran spasial dan komposisional tetap menjadi titik buta dalam benchmark yang ada. Kami menambahkan [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/), and [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Untuk tim yang menjalankan pipeline evaluasi jarak jauh, kami memperkenalkan server eval HTTP (#972). Bagi mereka yang membutuhkan ketelitian statistik, kami menambahkan CLT dan estimasi kesalahan standar terklaster (#989).

**Oktober 2025 (v0.5)** - Audio telah menjadi celah. Model bisa mendengar, tetapi kami tidak memiliki cara yang konsisten untuk mengujinya. Rilis ini menambahkan evaluasi audio komprehensif, caching respons untuk efisiensi, dan 50+ varian benchmark yang mencakup audio, visi, dan penalaran. [Catatan rilis](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md).

<details>
<summary>Di bawah ini adalah daftar kronologis tugas, model, dan fitur terbaru yang ditambahkan oleh kontributor luar biasa kami. </summary>

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** hadir! Rilis utama ini memperkenalkan evaluasi audio komprehensif, caching respons, 5 model baru (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), dan 50+ varian benchmark baru yang mencakup audio (Step2, VoiceBench, WenetSpeech), visi (CharXiv, Lemonade), dan penalaran (CSBench, SciBench, MedQA, SuperGPQA). Lihat [catatan rilis](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) untuk detail.
- [2025-07] 🚀🚀 Kami telah merilis `lmms-eval-0.4`. Lihat [catatan rilis](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) untuk detail lebih lanjut.

</details>

## Mengapa `lmms-eval`?

Kita sedang dalam perjalanan yang menarik menuju penciptaan Kecerdasan Buatan Umum (AGI), mirip dengan antusiasme pendaratan di bulan tahun 1960-an. Perjalanan ini didorong oleh model bahasa besar yang canggih (LLMs) dan model multimodal besar (LMMs), sistem kompleks yang mampu memahami, belajar, dan melakukan berbagai tugas manusia.

Tetapi inilah masalahnya: sistem pengukuran kami belum sejalan dengan ambisi kami.

Kami memiliki benchmark - ratusan jumlahnya. Tetapi mereka tersebar di folder Google Drive, tautan Dropbox, situs web universitas, dan server lab. Setiap benchmark memiliki format datanya sendiri, skrip evaluasinya sendiri, keunikannya sendiri. Ketika dua tim melaporkan hasil pada benchmark yang sama, mereka sering mendapatkan angka yang berbeda. Bukan karena model mereka berbeda, tetapi karena pipeline evaluasi mereka berbeda.

Bayangkan jika, selama perlombaan ruang angkasa, setiap negara mengukur jarak dalam unit yang berbeda dan tidak pernah membagikan tabel konversi mereka. Itulah kira-kira posisi kita saat ini dengan evaluasi multimodal.

Ini bukan sekadar ketidaknyamanan kecil. Ini adalah kegagalan sistemik. Tanpa pengukuran yang konsisten, kita tidak dapat mengetahui model mana yang sebenarnya lebih baik. Kita tidak dapat mereproduksi hasil. Kita tidak dapat membangun karya satu sama lain.

Untuk model bahasa, masalah ini sebagian besar diselesaikan oleh [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Ini menyediakan pemuatan data yang terpadu, evaluasi yang terstandarisasi, dan hasil yang dapat direproduksi. Ini mendukung [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). Ini telah menjadi infrastruktur.

Kami membangun `lmms-eval` untuk melakukan hal yang sama bagi model multimodal. Prinsip yang sama: satu kerangka kerja, antarmuka yang konsisten, angka yang dapat direproduksi. Moonshot membutuhkan penggaris yang andal.

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

**Evaluasi LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Evaluasi LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluasi Qwen2.5-VL**

```bash
bash examples/models/qwen2_5_vl.sh
```

**Evaluasi dengan tensor parallel untuk model yang lebih besar (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluasi dengan SGLang untuk model yang lebih besar (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Parameter Lainnya**

```bash
python3 -m lmms_eval --help
```

**Variabel Lingkungan**
Sebelum menjalankan eksperimen dan evaluasi, kami menyarankan Anda untuk mengekspor variabel lingkungan berikut ke lingkungan Anda. Beberapa diperlukan agar tugas tertentu dapat dijalankan.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Variabel lingkungan lain yang mungkin termasuk 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, dll.
```

**Masalah Lingkungan Umum**

Terkadang Anda mungkin menghadapi beberapa masalah umum, misalnya kesalahan yang terkait dengan httpx atau protobuf. Untuk mengatasi masalah ini, Anda dapat mencoba terlebih dahulu:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Jika Anda menggunakan numpy==2.x, terkadang dapat menyebabkan kesalahan
python3 -m pip install numpy==1.26;
# Terkadang sentencepiece diperlukan agar tokenizer dapat berfungsi
python3 -m pip install sentencepiece;
```

## Menambahkan Model dan Dataset Kustom

Lihat [dokumentasi](../README.md) kami.

## Ucapan Terima Kasih

lmms_eval adalah fork dari [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Kami menyarankan untuk membaca [dokumentasi lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) untuk informasi yang relevan.

---

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
