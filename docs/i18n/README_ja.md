<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
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

📖 [サポートタスク (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [サポートモデル (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [ドキュメント](../README.md)

---

## お知らせ

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** リリース！このメジャーリリースでは、包括的な音声評価、レスポンスキャッシング、5つの新モデル（GPT-4o Audio Preview、Gemma-3、LongViLA-R1、LLaVA-OneVision 1.5、Thyme）、および50以上の新しいベンチマークバリアント（音声：Step2、VoiceBench、WenetSpeech、視覚：CharXiv、Lemonade、推論：CSBench、SciBench、MedQA、SuperGPQA）を導入しています。詳細は[リリースノート](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md)をご覧ください。
- [2025-07] 🚀🚀 `lmms-eval-0.4` をリリースしました。詳細は[リリースノート](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)をご覧ください。

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

**その他のパラメータ**

```bash
python3 -m lmms_eval --help
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
```
