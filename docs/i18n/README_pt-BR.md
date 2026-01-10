<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Suite de Avaliação de Grandes Modelos Multimodais

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | **Português** | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Acelerando o desenvolvimento de grandes modelos multimodais (LMMs) com `lmms-eval`. Suportamos a maioria das tarefas de texto, imagem, vídeo e áudio.

🏠 [Página Inicial LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Conjuntos de Dados Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Tarefas Suportadas (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Modelos Suportados (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentação](../README.md)

---

## Anúncios

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** está aqui! Esta versão principal introduz avaliação de áudio abrangente, cache de respostas, 5 novos modelos (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), e mais de 50 novas variantes de benchmark abrangendo áudio (Step2, VoiceBench, WenetSpeech), visão (CharXiv, Lemonade) e raciocínio (CSBench, SciBench, MedQA, SuperGPQA). Consulte as [notas de lançamento](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) para detalhes.
- [2025-07] 🚀🚀 Lançamos `lmms-eval-0.4`. Consulte as [notas de lançamento](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) para mais detalhes.

## Por que `lmms-eval`?

Estamos em uma jornada emocionante em direção à criação da Inteligência Artificial Geral (AGI), semelhante ao entusiasmo da alunissagem dos anos 1960. Esta jornada é impulsionada por modelos de linguagem avançados (LLMs) e grandes modelos multimodais (LMMs), sistemas complexos capazes de entender, aprender e executar uma ampla variedade de tarefas humanas.

Para medir o quão avançados esses modelos são, usamos uma variedade de benchmarks de avaliação. Esses benchmarks são ferramentas que nos ajudam a entender as capacidades desses modelos, mostrando-nos o quão perto estamos de alcançar AGI. No entanto, encontrar e usar esses benchmarks é um grande desafio.

No campo dos modelos de linguagem, o trabalho de [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) estabeleceu um precedente valioso. Absorvemos o design requintado e eficiente do lm-evaluation-harness e introduzimos o **lmms-eval**, um framework de avaliação meticulosamente elaborado para avaliação consistente e eficiente de LMM.

## Instalação

### Usando uv (Recomendado para ambientes consistentes)

Usamos `uv` para gerenciamento de pacotes para garantir que todos os desenvolvedores usem exatamente as mesmas versões de pacotes. Primeiro, instale o uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para desenvolvimento com ambiente consistente:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Recomendado
uv pip install -e ".[all]"
# Se você quiser usar uv sync
# uv sync  # Isso cria/atualiza seu ambiente a partir de uv.lock
```

Para executar comandos:
```bash
uv run python -m lmms_eval --help  # Executar qualquer comando com uv run
```

### Instalação Alternativa

Para uso direto do Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Você pode precisar adicionar e incluir seu próprio yaml de tarefas se usar esta instalação
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Uso

> Mais exemplos em [examples/models](../../examples/models)

**Avaliação de Modelo Compatível com OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Avaliação de vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Avaliação de LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Mais Parâmetros**

```bash
python3 -m lmms_eval --help
```

## Adicionar Modelo e Conjunto de Dados Personalizados

Consulte nossa [documentação](../README.md).

## Agradecimentos

lmms_eval é um fork de [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Recomendamos ler a [documentação do lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) para informações relevantes.

## Citações

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
