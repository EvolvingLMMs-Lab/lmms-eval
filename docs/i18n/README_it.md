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

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** è qui! Questa versione principale introduce valutazione audio completa, caching delle risposte, 5 nuovi modelli (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), e oltre 50 nuove varianti di benchmark che coprono audio (Step2, VoiceBench, WenetSpeech), visione (CharXiv, Lemonade) e ragionamento (CSBench, SciBench, MedQA, SuperGPQA). Consulta le [note di rilascio](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) per i dettagli.
- [2025-07] 🚀🚀 Abbiamo rilasciato `lmms-eval-0.4`. Consulta le [note di rilascio](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) per maggiori dettagli.

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

**Altri Parametri**

```bash
python3 -m lmms_eval --help
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
```
