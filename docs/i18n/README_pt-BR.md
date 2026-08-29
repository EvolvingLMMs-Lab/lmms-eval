<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
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

📖 [Tarefas Suportadas (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [Modelos Suportados (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentação](../README.md)

---

## O que há de novo

Avaliar modelos multimodais é mais difícil do que parece. Temos centenas de benchmarks, mas nenhuma forma padronizada de executá-los. Os resultados variam entre laboratórios. As comparações tornam-se não confiáveis. Temos trabalhado para resolver isso - não através de um esforço heróico, mas através de um processo sistemático.

**Janeiro de 2026** - Reconhecemos que o raciocínio espacial e composicional permaneciam pontos cegos nos benchmarks existentes. Adicionamos [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/) e [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Para equipes que executam pipelines de avaliação remota, introduzimos um servidor de avaliação HTTP (#972). Para aqueles que precisam de rigor estatístico, adicionamos CLT e estimativa de erro padrão agrupado (#989).

**Outubro de 2025 (v0.5)** - O áudio era uma lacuna. Os modelos podiam ouvir, mas não tínhamos uma forma consistente de testá-los. Este lançamento adicionou avaliação de áudio abrangente, cache de respostas para eficiência e mais de 50 variantes de benchmarks abrangendo áudio, visão e raciocínio. [Notas de lançamento](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md).

<details>
<summary>Abaixo está uma lista cronológica de tarefas, modelos e recursos recentes adicionados pelos nossos incríveis colaboradores. </summary>

- [2025-01] 🎓🎓 Lançamos nosso novo benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Consulte a [página do projeto](https://videommmu.github.io/) para mais detalhes.
- [2024-12] 🎉🎉 Apresentamos o [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), juntamente com a [Equipe MME](https://github.com/BradyFU/Video-MME) e a [Equipe OpenCompass](https://github.com/open-compass).
- [2024-11] 🔈🔊 O `lmms-eval/v0.3.0` foi atualizado para suportar avaliações de áudio para modelos de áudio como Qwen2-Audio e Gemini-Audio em tarefas como AIR-Bench, Clotho-AQA, LibriSpeech e muito mais. Consulte o [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.3.md) para mais detalhes!
- [2024-10] 🎉🎉 Damos as boas-vindas à nova tarefa [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), um benchmark VQA focado em visão (NeurIPS'24) que desafia modelos de visão e linguagem com perguntas simples sobre imagens naturais.
- [2024-10] 🎉🎉 Damos as boas-vindas à nova tarefa [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) para compreensão temporal detalhada e raciocínio para vídeos, que revela uma enorme lacuna de mais de 30% entre humanos e IA.
- [2024-10] 🎉🎉 Damos as boas-vindas às novas tarefas [VDC](https://rese1f.github.io/aurora-web/) para legendagem detalhada de vídeo, [MovieChat-1K](https://rese1f.github.io/MovieChat/) para compreensão de vídeo de formato longo e [Vinoground](https://vinoground.github.io/), um benchmark LMM temporal contrafactual composto por 1000 pares curtos de vídeo-legenda naturais. Também damos as boas-vindas aos novos modelos: [AuroraCap](https://github.com/rese1f/aurora) e [MovieChat](https://github.com/rese1f/MovieChat).
- [2024-09] 🎉🎉 Damos as boas-vindas às novas tarefas [MMSearch](https://mmsearch.github.io/) e [MME-RealWorld](https://mme-realworld.github.io/) para aceleração de inferência.
- [2024-09] ⚙️️⚙️️️️ Atualizamos o `lmms-eval` para `0.2.3` com mais tarefas e recursos. Suportamos um conjunto compacto de avaliações de tarefas de linguagem (crédito de código para [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), e removemos a lógica de registro no início (para todos os modelos e tarefas) para reduzir a sobrecarga. Agora o `lmms-eval` lança apenas as tarefas/modelos necessários. Verifique as [notas de lançamento](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) para mais detalhes.
- [2024-08] 🎉🎉 Damos as boas-vindas ao novo modelo [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), novas tarefas [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). Fornecemos o novo recurso de SGlang Runtime API para o modelo llava-onevision, consulte o [documento](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/getting-started/commands.md) para aceleração de inferência.
- [2024-07] 👨‍💻👨‍💻 O `lmms-eval/v0.2.1` foi atualizado para suportar mais modelos, incluindo [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA) e muitas outras tarefas de avaliação, por exemplo, [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) e [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 Lançamos o [relatório técnico](https://arxiv.org/abs/2407.12772) e o [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)! 
- [2024-06] 🎬🎬 O `lmms-eval/v0.2.0` foi atualizado para suportar avaliações de vídeo para modelos de vídeo como LLaVA-NeXT Video e Gemini 1.5 Pro em tarefas como EgoSchema, PerceptionTest, VideoMME e muito mais. Consulte o [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) para mais detalhes!
- [2024-03] 📝📝 Lançamos a primeira versão do `lmms-eval`, consulte o [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) para mais detalhes!

</details>

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

**Avaliação de LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Avaliação de LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Avaliação de Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Avaliação de LLaVA no MME**

Se você quiser testar o LLaVA 1.5, você terá que clonar o repositório deles de [LLaVA](https://github.com/haotian-liu/LLaVA) e

```bash
bash examples/models/llava_next.sh
```

**Avaliação com paralelismo de tensores para modelos maiores (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Avaliação com SGLang para modelos maiores (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Avaliação com vLLM para modelos maiores (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Mais Parâmetros**

```bash
python3 -m lmms_eval --help
```

**Variáveis de Ambiente**
Antes de executar experimentos e avaliações, recomendamos exportar as seguintes variáveis de ambiente. Algumas são necessárias para a execução de certas tarefas.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Outras possíveis variáveis de ambiente incluem 
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY etc.
```

**Problemas Comuns de Ambiente**

Às vezes, você pode encontrar problemas comuns, por exemplo, erros relacionados ao httpx ou protobuf. Para resolver esses problemas, você pode tentar primeiro:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# Se você estiver usando numpy==2.x, às vezes pode causar erros
python3 -m pip install numpy==1.26;
# Às vezes, sentencepiece é necessário para o tokenizer funcionar
python3 -m pip install sentencepiece;
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
