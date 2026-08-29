<p align="center">
  <img src="../images/lmms-eval-hero.avif" alt="LMMs-Eval" width="70%">
</p>

# 大規模マルチモーダルモデル評価スイート

🌐 [English](../../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | **日本語** | [한국어](README_ko.md) | [Español](README_es.md) | [Français](README_fr.md) | [Deutsch](README_de.md) | [Português](README_pt-BR.md) | [Русский](README_ru.md) | [Italiano](README_it.md) | [Nederlands](README_nl.md) | [Polski](README_pl.md) | [Türkçe](README_tr.md) | [العربية](README_ar.md) | [हिन्दी](README_hi.md) | [Tiếng Việt](README_vi.md) | [Indonesia](README_id.md)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> `lmms-eval` で大規模マルチモーダルモデル（LMMs）の開発を加速。テキスト、画像、ビデオ、オーディオのタスクをサポートしています。

🏠 [LMMs-Lab ホームページ](https://www.lmms-lab.com/) | 🤗 [Huggingface データセット](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [サポートタスク (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [サポートモデル (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [ドキュメント](../README.md)

---

## お知らせ

マルチモーダルモデルの評価は、見た目以上に困難です。何百ものベンチマークが存在しますが、それらを実行するための標準的な方法はありません。ラボ間で結果が異なり、比較の信頼性が低下します。私たちは、個々の努力ではなく、体系的なプロセスを通じてこの問題の解決に取り組んできました。

**2026年1月** - 既存のベンチマークにおいて空間的および構成的な推論が依然として盲点であることを認識しました。[CaptionQA](https://captionqa.github.io/)、[SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench)、[SiteBench](https://sitebench.github.io/)、[ViewSpatial](https://github.com/ViewSpatial/ViewSpatial) を追加しました。リモート評価パイプラインを実行するチームのために、HTTP評価サーバー（#972）を導入しました。また、統計的な厳密さを必要とする方のために、CLT（中心極限定理）とクラスター化標準誤差の推定（#989）を追加しました。

**2025年10月 (v0.5)** - 音声評価が課題となっていました。モデルは音声を認識できましたが、それをテストするための一貫した方法がありませんでした。このリリースでは、包括的な音声評価、効率のためのレスポンスキャッシング、および音声、視覚、推論にわたる 50 以上のベンチマークバリアントが導入されました。[リリースノート](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.5.md)。

<details>
<summary>以下は、素晴らしいコントリビューターによって追加された最近のタスク、モデル、および機能の時系列リストです。</summary>

- [2025-01] 🎓🎓 新しいベンチマークをリリースしました：[Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826)。詳細は[プロジェクトページ](https://videommmu.github.io/)を参照してください。
- [2025-07] 🚀🚀 `lmms-eval-0.4` をリリースしました。詳細は[リリースノート](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/releases/lmms-eval-0.4.md)をご覧ください。

</details>

## なぜ `lmms-eval` なのか？

私たちは、1960年代の月面着陸のような熱意を持って、人工汎用知能（AGI）の創造に向けたエキサイティングな旅を進めています。この旅は、人間のさまざまなタスクを理解、学習、実行できる複雑なシステムである、高度な大規模言語モデル（LLMs）と大規模マルチモーダルモデル（LMMs）によって推進されています。

これらのモデルがどれほど高度であるかを測定するために、さまざまな評価ベンチマークを使用します。これらのベンチマークは、これらのモデルの能力を理解し、AGIの達成にどれだけ近づいているかを示すツールです。しかし、これらのベンチマークを見つけて使用することは大きな課題です。

言語モデルの分野では、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) の先例が貴重な道標となっています。私たちは lm-evaluation-harness の精巧で効率的なデザインを吸収し、LMMの一貫した効率的な評価のために丹念に作られた評価フレームワーク **lmms-eval** を導入しました。

## インストール

### uv の使用（一貫した環境に推奨）

すべての開発者がまったく同じパッケージバージョンを使用できるように、`uv` をパッケージ管理に使用しています。まず、uv をインストールしてください：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

一貫した環境での開発：
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# 推奨
uv pip install -e ".[all]"
# uv sync を使用したい場合
# uv sync  # これは uv.lock から環境を作成/更新します
```

コマンドの実行：
```bash
uv run python -m lmms_eval --help  # uv run で任意のコマンドを実行
```

### 代替インストール方法

Git からの直接使用：
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# このインストール方法を使用する場合、独自のタスク yaml を追加してインクルードする必要があるかもしれません
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## 使用方法

> 詳細な例は [examples/models](../../examples/models) を参照してください

**OpenAI互換モデルの評価**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM の評価**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision の評価**

```bash
bash examples/models/llava_onevision.sh
```

**LLaVA-OneVision1_5 の評価**

```bash
bash examples/models/llava_onevision1_5.sh
```

**LLaMA-3.2-Vision の評価**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2.5-VL の評価**

```bash
bash examples/models/qwen2_5_vl.sh
```

**大きなモデル（llava-next-72b）のテンソル並列による評価**

```bash
bash examples/models/tensor_parallel.sh
```

**大きなモデル（llava-next-72b）の SGLang による評価**

```bash
bash examples/models/sglang.sh
```

**その他のパラメータ**

```bash
python3 -m lmms_eval --help
```

**環境変数**

実験や評価を実行する前に、以下の環境変数をエクスポートすることをお勧めします。一部の変数は特定のタスクの実行に必要です。

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# その他の利用可能な環境変数には以下が含まれます
# ANTHROPIC_API_KEY, DASHSCOPE_API_KEY など
```

**よくある環境の問題**

httpx や protobuf に関連するエラーなど、いくつかの一般的な問題が発生する場合があります。これらの問題を解決するには、まず以下を試してください。

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# numpy==2.x を使用している場合、エラーが発生することがあります
python3 -m pip install numpy==1.26;
# トークナイザーの動作に sentencepiece が必要な場合があります
python3 -m pip install sentencepiece;
```

## カスタムモデルとデータセットの追加

[ドキュメント](../README.md)を参照してください。

## 謝辞

lmms_eval は [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) のフォークです。関連情報については lm-eval-harness の[ドキュメント](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)をお読みになることをお勧めします。

## 引用

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
