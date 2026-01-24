<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Suite di Valutazione per Grandi Modelli Multimodali

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | **Italiano** | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Accelerare lo sviluppo di grandi modelli multimodali (LMMs) con `lmms-eval`. Supportiamo la maggior parte delle attività di testo, immagine, video e audio.

🏠 [Home Page LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Dataset Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Attività Supportate (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Modelli Supportati (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentazione](../README.md)

---

## Annunci

Valutare i modelli multimodali è più difficile di quanto sembri. Abbiamo centinaia di benchmark, ma nessun modo standard per eseguirli. I risultati variano tra i laboratori. I confronti diventano inaffidabili. Abbiamo lavorato per affrontare questo problema - non attraverso uno sforzo eroico, ma attraverso un processo sistematico.

**Gennaio 2026** - Abbiamo riconosciuto che il ragionamento spaziale e compositivo rimanevano punti ciechi nei benchmark esistenti. Abbiamo aggiunto [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/) e [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Per i team che eseguono pipeline di valutazione remota, abbiamo introdotto un server di valutazione HTTP (#972). Per coloro che necessitano di rigore statistico, abbiamo aggiunto CLT e la stima dell'errore standard raggruppato (#989).

**Ottobre 2025 (v0.5)** - L'audio era stato una lacuna. I modelli potevano sentire, ma non avevamo un modo coerente per testarli. Questa versione ha aggiunto una valutazione audio completa, il caching delle risposte per l'efficienza e oltre 50 varianti di benchmark che spaziano tra audio, visione e ragionamento. [Note di rilascio](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md).

## Perché `lmms-eval`?

Siamo in un viaggio entusiasmante verso la creazione dell'Intelligenza Artificiale Generale (AGI), simile all'entusiasmo dell'allunaggio degli anni '60. Questo viaggio è alimentato da modelli linguistici avanzati (LLMs) e grandi modelli multimodali (LMMs), sistemi complessi capaci di comprendere, apprendere e svolgere un'ampia varietà di compiti umani.

Per misurare quanto sono avanzati questi modelli, utilizziamo una varietà di benchmark di valutazione. Questi benchmark sono strumenti che ci aiutano a comprendere le capacità di questi modelli, mostrandoci quanto siamo vicini al raggiungimento dell'AGI. Tuttavia, trovare e utilizzare questi benchmark è una grande sfida.

Nel campo dei modelli linguistici, il lavoro di [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ha stabilito un prezioso precedente. Abbiamo assorbito il design squisito ed efficiente di lm-evaluation-harness e introdotto **lmms-eval**, un framework di valutazione meticolosamente realizzato per una valutazione coerente ed efficiente degli LMM.

## Installazione

### Utilizzando uv (Raccomandato per ambienti coerenti)

Utilizziamo `uv` per la gestione dei pacchetti per garantire che tutti gli sviluppatori utilizzino esattamente le stesse versioni dei pacchetti. Prima, installa uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Per lo sviluppo con ambiente coerente:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Raccomandato
uv pip install -e ".[all]"
# Se vuoi usare uv sync
# uv sync  # Questo crea/aggiorna il tuo ambiente da uv.lock
```

Per eseguire comandi:
```bash
uv run python -m lmms_eval --help  # Eseguire qualsiasi comando con uv run
```

### Installazione Alternativa

Per uso diretto da Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Potresti dover aggiungere e includere il tuo yaml delle attività se usi questa installazione
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Utilizzo

> Altri esempi in [examples/models](../../examples/models)

**Valutazione di Modello Compatibile con OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Valutazione di vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Valutazione di LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Valutazione di LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Valutazione di LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Valutazione di Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Valutazione con tensor parallel per modelli più grandi (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Valutazione con SGLang per modelli più grandi (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Altri Parametri**

```bash
python3 -m lmms_eval --help
```

**Variabili d'ambiente**
Prima di eseguire esperimenti e valutazioni, ti consigliamo di esportare le seguenti variabili d'ambiente nel tuo ambiente. Alcune sono necessarie per l'esecuzione di determinate attività.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Altre possibili variabili d'ambiente includono 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY ecc.
```

**Problemi comuni dell'ambiente**

A volte potresti riscontrare alcuni problemi comuni, ad esempio errori relativi a httpx o protobuf. Per risolvere questi problemi, puoi prima provare:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Se stai usando numpy==2.x, a volte può causare errori
python3 -m pip install numpy==1.26;
# A volte sentencepiece è necessario per il funzionamento del tokenizer
python3 -m pip install sentencepiece;
```

## Aggiungere Modello e Dataset Personalizzati

Consulta la nostra [documentazione](../README.md).

## Riconoscimenti

lmms_eval è un fork di [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Consigliamo di leggere la [documentazione di lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) per informazioni rilevanti.

## Citazioni

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
