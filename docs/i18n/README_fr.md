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

## Quoi de Neuf

Évaluer des modèles multimodaux est plus difficile qu'il n'y paraît. Nous disposons de centaines de benchmarks, mais d'aucune méthode standard pour les exécuter. Les résultats varient d'un laboratoire à l'autre. Les comparaisons deviennent peu fiables. Nous nous efforçons de remédier à ce problème - non par un effort héroïque, mais par un processus systématique.

**Janvier 2026** - Nous avons reconnu que le raisonnement spatial et compositionnel restait un angle mort dans les benchmarks existants. Nous avons ajouté [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/), et [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). Pour les équipes gérant des pipelines d'évaluation à distance, nous avons introduit un serveur d'évaluation HTTP (#972). Pour ceux qui ont besoin de rigueur statistique, nous avons ajouté le CLT (théorème central limite) et l'estimation de l'erreur standard groupée (#989).

**Octobre 2025 (v0.5)** - L'audio était une lacune. Les modèles pouvaient entendre, mais nous n'avions aucun moyen cohérent de les tester. Cette version a ajouté une évaluation audio complète, la mise en cache des réponses pour plus d'efficacité, et plus de 50 variantes de benchmarks couvrant l'audio, la vision et le raisonnement. [Notes de version](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md).

<details>
<summary>Ci-dessous une liste chronologique des tâches, modèles et fonctionnalités récents ajoutés par nos incroyables contributeurs. </summary>

- [2025-01] 🎓🎓 Nous avons publié notre nouveau benchmark : [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Veuillez vous référer à la [page du projet](https://videommmu.github.io/) pour plus de détails.
- [2024-12] 🎉🎉 Nous avons présenté [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), conjointement avec l'[Équipe MME](https://github.com/BradyFU/Video-MME) et l'[Équipe OpenCompass](https://github.com/open-compass).
- [2024-11] 🔈🔊 `lmms-eval/v0.3.0` a été mis à jour pour supporter les évaluations audio pour des modèles audio comme Qwen2-Audio et Gemini-Audio sur des tâches telles que AIR-Bench, Clotho-AQA, LibriSpeech, et plus encore. Veuillez vous référer au [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) pour plus de détails !
- [2024-10] 🎉🎉 Nous accueillons la nouvelle tâche [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), un benchmark VQA centré sur la vision (NeurIPS'24) qui défie les modèles vision-langage avec des questions simples sur l'imagerie naturelle.
- [2024-10] 🎉🎉 Nous accueillons la nouvelle tâche [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) pour une compréhension temporelle fine et un raisonnement pour les vidéos, qui révèle un écart énorme (>30%) entre l'humain et l'IA.
- [2024-10] 🎉🎉 Nous accueillons les nouvelles tâches [VDC](https://rese1f.github.io/aurora-web/) pour le légendage détaillé de vidéos, [MovieChat-1K](https://rese1f.github.io/MovieChat/) pour la compréhension de vidéos longue durée, et [Vinoground](https://vinoground.github.io/), un benchmark LMM temporel contrefactuel composé de 1000 paires courtes de vidéos-légendes naturelles. Nous accueillons également les nouveaux modèles : [AuroraCap](https://github.com/rese1f/aurora) et [MovieChat](https://github.com/rese1f/MovieChat).
- [2024-09] 🎉🎉 Nous accueillons les nouvelles tâches [MMSearch](https://mmsearch.github.io/) et [MME-RealWorld](https://mme-realworld.github.io/) pour l'accélération de l'inférence.
- [2024-09] ⚙️️⚙️️️️ Nous mettons à jour `lmms-eval` vers `0.2.3` avec plus de tâches et de fonctionnalités. Nous supportons un ensemble compact d'évaluations de tâches de langage (crédit code à [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), et nous supprimons la logique d'enregistrement au démarrage (pour tous les modèles et tâches) pour réduire la surcharge. Désormais, `lmms-eval` ne lance que les tâches/modèles nécessaires. Veuillez consulter les [notes de version](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) pour plus de détails.
- [2024-08] 🎉🎉 Nous accueillons le nouveau modèle [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), les nouvelles tâches [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). Nous fournissons une nouvelle fonctionnalité d'API SGlang Runtime pour le modèle llava-onevision, veuillez vous référer au [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) pour l'accélération de l'inférence.
- [2024-07] 👨‍💻👨‍💻 `lmms-eval/v0.2.1` a été mis à jour pour supporter plus de modèles, incluant [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), et bien d'autres tâches d'évaluation, par exemple [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) et [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 Nous avons publié le [rapport technique](https://arxiv.org/abs/2407.12772) et [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench) !
- [2024-06] 🎬🎬 `lmms-eval/v0.2.0` a été mis à jour pour supporter les évaluations vidéo pour des modèles vidéo comme LLaVA-NeXT Video et Gemini 1.5 Pro sur des tâches telles que EgoSchema, PerceptionTest, VideoMME, et plus encore. Veuillez vous référer au [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) pour plus de détails !
- [2024-03] 📝📝 Nous avons publié la première version de `lmms-eval`, veuillez vous référer au [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) pour plus de détails !

</details>

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

**Évaluation de LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Évaluation de LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Évaluation de Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Évaluation de LLaVA sur MME**

Si vous voulez tester LLaVA 1.5, vous devrez cloner leur dépôt depuis [LLaVA](https://github.com/haotian-liu/LLaVA) et

```bash
bash examples/models/llava_next.sh
```

**Évaluation avec parallélisme de tenseurs pour les modèles plus grands (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Évaluation avec SGLang pour les modèles plus grands (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Évaluation avec vLLM pour les modèles plus grands (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Plus de Paramètres**

```bash
python3 -m lmms_eval --help
```

**Variables d'Environnement**
Avant d'exécuter des expériences et des évaluations, nous vous recommandons d'exporter les variables d'environnement suivantes dans votre environnement. Certaines sont nécessaires pour l'exécution de certaines tâches.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Problèmes d'Environnement Courants**

Parfois, vous pourriez rencontrer des problèmes courants, par exemple des erreurs liées à httpx ou protobuf. Pour résoudre ces problèmes, vous pouvez d'abord essayer :

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
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

@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}
``````
