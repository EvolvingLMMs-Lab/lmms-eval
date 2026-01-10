<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# Suite d'Évaluation des Grands Modèles Multimodaux

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Español](README_es.md) | **Français** | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Accélérer le développement des grands modèles multimodaux (LMMs) avec `lmms-eval`. Nous supportons la plupart des tâches de texte, d'image, de vidéo et d'audio.

🏠 [Page d'Accueil LMMs-Lab](https://www.lmms-lab.com/) | 🤗 [Jeux de Données Huggingface](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Tâches Supportées (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Modèles Supportés (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](../README.md)

---

## Annonces

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** est là ! Cette version majeure introduit une évaluation audio complète, la mise en cache des réponses, 5 nouveaux modèles (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), et plus de 50 nouvelles variantes de benchmark couvrant l'audio (Step2, VoiceBench, WenetSpeech), la vision (CharXiv, Lemonade) et le raisonnement (CSBench, SciBench, MedQA, SuperGPQA). Consultez les [notes de version](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) pour plus de détails.
- [2025-07] 🚀🚀 Nous avons publié `lmms-eval-0.4`. Consultez les [notes de version](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) pour plus de détails.

## Pourquoi `lmms-eval` ?

Nous sommes dans un voyage passionnant vers la création de l'Intelligence Artificielle Générale (AGI), similaire à l'enthousiasme de l'alunissage des années 1960. Ce voyage est propulsé par des modèles de langage avancés (LLMs) et des grands modèles multimodaux (LMMs), des systèmes complexes capables de comprendre, d'apprendre et d'effectuer une grande variété de tâches humaines.

Pour mesurer l'avancement de ces modèles, nous utilisons une variété de benchmarks d'évaluation. Ces benchmarks sont des outils qui nous aident à comprendre les capacités de ces modèles, nous montrant à quel point nous sommes proches d'atteindre l'AGI. Cependant, trouver et utiliser ces benchmarks est un défi majeur.

Dans le domaine des modèles de langage, le travail de [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) a établi un précédent précieux. Nous avons absorbé la conception exquise et efficace de lm-evaluation-harness et introduit **lmms-eval**, un framework d'évaluation méticuleusement conçu pour une évaluation cohérente et efficace des LMM.

## Installation

### Utilisation de uv (Recommandé pour des environnements cohérents)

Nous utilisons `uv` pour la gestion des paquets afin de garantir que tous les développeurs utilisent exactement les mêmes versions de paquets. Tout d'abord, installez uv :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Pour le développement avec un environnement cohérent :
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Recommandé
uv pip install -e ".[all]"
# Si vous voulez utiliser uv sync
# uv sync  # Ceci crée/met à jour votre environnement depuis uv.lock
```

Pour exécuter des commandes :
```bash
uv run python -m lmms_eval --help  # Exécuter n'importe quelle commande avec uv run
```

### Installation Alternative

Pour une utilisation directe depuis Git :
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# Vous devrez peut-être ajouter et inclure votre propre yaml de tâches si vous utilisez cette installation
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Utilisation

> Plus d'exemples dans [examples/models](../../examples/models)

**Évaluation de Modèle Compatible OpenAI**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Évaluation de vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Évaluation de LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Plus de Paramètres**

```bash
python3 -m lmms_eval --help
```

## Ajouter un Modèle et un Jeu de Données Personnalisés

Consultez notre [documentation](../README.md).

## Remerciements

lmms_eval est un fork de [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Nous vous recommandons de lire la [documentation de lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) pour des informations pertinentes.

## Citations

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
