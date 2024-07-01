<p align="center" width="80%">
<img src="https://i.postimg.cc/g0QRgMVv/WX20240228-113337-2x.png"  width="100%" height="70%">
</p>

# The Evaluation Suite of Large Multimodal Models 

> Accelerating the development of large multimodal models (LMMs) with `lmms-eval`

🏠 [LMMs-Lab Homepage](https://lmms-lab.github.io/) |  🎉 [Blog](https://lmms-lab.github.io/lmms-eval-blog/lmms-eval-0.1/) | 📚 [Documentation](docs/README.md) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

---

## Annoucement

- [2024-06] 🎬🎬 The `lmms-eval/v0.2` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details

- [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details

## Why `lmms-eval`?

<p align="center" width="80%">
<img src="https://i.postimg.cc/L5kNJsJf/Blue-Purple-Futuristic-Modern-3-D-Tech-Company-Business-Presentation.png"  width="100%" height="80%">
</p>

In today's world, we're on an exciting journey toward creating Artificial General Intelligence (AGI), much like the enthusiasm of the 1960s moon landing. This journey is powered by advanced large language models (LLMs) and large multimodal models (LMMs), which are complex systems capable of understanding, learning, and performing a wide variety of human tasks.

To gauge how advanced these models are, we use a variety of evaluation benchmarks. These benchmarks are tools that help us understand the capabilities of these models, showing us how close we are to achieving AGI. 

However, finding and using these benchmarks is a big challenge. The necessary benchmarks and datasets are spread out and hidden in various places like Google Drive, Dropbox, and different school and research lab websites. It feels like we're on a treasure hunt, but the maps are scattered everywhere.

In the field of language models, there has been a valuable precedent set by the work of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). They offer integrated data and model interfaces, enabling rapid evaluation of language models and serving as the backend support framework for the [open-llm-leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and has gradually become the underlying ecosystem of the era of foundation models.

We humbly obsorbed the exquisite and efficient design of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and introduce **lmms-eval**, an evaluation framework meticulously crafted for consistent and efficient evaluation of LMM.

## Installation

For formal usage, you can install the package from PyPI by running the following command:
```bash
pip install lmms-eval
```

For development, you can install the package by cloning the repository and running the following command:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .
```

If you wanted to test llava, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and
```bash
# for llava 1.5
# git clone https://github.com/haotian-liu/LLaVA
# cd LLaVA
# pip install -e .

# for llava-next (1.6)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e .
```

<details>
<summary>Reproduction of LLaVA-1.5's paper results</summary>

You can check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to **reproduce LLaVA-1.5's paper results**. We found torch/cuda versions difference would cause small variations in the results, we provide the [results check](miscs/llava_result_check.md) with different environments.

</details>

If you want to test on caption dataset such as `coco`, `refcoco`, and `nocaps`, you will need to have `java==1.8.0 ` to let pycocoeval api to work. If you don't have it, you can install by using conda
```
conda install openjdk=8
```
you can then check your java version by `java -version` 


<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>


Our Development will be continuing on the main branch, and we encourage you to give us feedback on what features are desired and how to improve the library further, or ask questions, either in issues or PRs on GitHub.

## Multiple Usages

**Evaluation of LLaVA on MME**

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path ./logs/
```

**Evaluation of LLaVA on multiple datasets**

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme,mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs/
```

**For other variants llava. Please change the `conv_template` in the `model_args`**

> `conv_template` is an arg of the init function of llava in `lmms_eval/models/llava.py`, you could find the corresponding value at LLaVA's code, probably in a dict variable `conv_templates` in `llava/conversations.py`

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-mistral-7b,conv_template=mistral_instruct" \
    --tasks mme,mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs/
```

**Evaluation of larger lmms (llava-v1.6-34b)**

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-34b,conv_template=mistral_direct" \
    --tasks mme,mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs/
```

**Evaluation with a set of configurations, supporting evaluation of multiple models and datasets**

```bash
python3 -m accelerate.commands.launch --num_processes=8 -m lmms_eval --config ./miscs/example_eval.yaml
```

**Evaluation with naive model sharding for bigger model (llava-next-72b)**

```bash
python3 -m lmms_eval \
    --model=llava \
    --model_args=pretrained=lmms-lab/llava-next-72b,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
    --tasks=pope,vizwiz_vqa_val,scienceqa_img \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix=llava_qwen \
    --output_path="./logs/" \
    --wandb_args=project=lmms-eval,job_type=eval,entity=llava-vl
```

**Evaluation with SGLang for bigger model (llava-next-72b)**

```bash
python3 -m lmms_eval \
	--model=llava_sglang \
	--model_args=pretrained=lmms-lab/llava-next-72b,tokenizer=lmms-lab/llavanext-qwen-tokenizer,conv_template=chatml-llava,tp_size=8,parallel=8 \
	--tasks=mme \
	--batch_size=1 \
	--log_samples \
	--log_samples_suffix=llava_qwen \
	--output_path=./logs/ \
	--verbosity=INFO
```

### Supported models

Please check [supported models](lmms_eval/models/__init__.py) for more details.

### Supported tasks

Please check [supported tasks](lmms_eval/docs/current_tasks.md) for more details.

## Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md).

## Acknowledgement

lmms_eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We recommend you to read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for relevant information. 

---

Below are the changes we made to the original API:
- Build context now only pass in idx and process image and doc during the model responding phase. This is due to the fact that dataset now contains lots of images and we can't store them in the doc like the original lm-eval-harness other wise the cpu memory would explode.
- Instance.args (lmms_eval/api/instance.py) now contains a list of images to be inputted to lmms.
- lm-eval-harness supports all HF language models as single model class. Currently this is not possible of lmms because the input/output format of lmms in HF are not yet unified. Thererfore, we have to create a new class for each lmms model. This is not ideal and we will try to unify them in the future.

---

During the initial stage of our project, we thank:
- [Xiang Yue](https://xiangyue9607.github.io/), [Jingkang Yang](https://jingkang50.github.io/), [Dong Guo](https://www.linkedin.com/in/dongguoset/) and [Sheng Shen](https://sincerass.github.io/) for early discussion and testing.

---

During the `v0.1` to `v0.2`, we thank the community support from pull requests (PRs):

> Details are in [lmms-eval/v0.2.0 release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/untagged-9057ff0e9a72d5a5846f)

**Datasets:**

- VCR: Visual Caption Restoration (officially from the authors, MILA)
- ConBench (officially from the authors, PKU/Bytedance)
- MathVerse (officially from the authors, CUHK)
- MM-UPD (officially from the authors, University of Tokyo)
- WebSRC (from Hunter Heiden)
- ScreeSpot (from Hunter Heiden)
- RealworldQA (from Fanyi Pu, NTU)
- Multi-lingual LLaVA-W (from Gagan Bhatia, UBC)

**Models:**

- LLaVA-HF (officially from Huggingface)
- Idefics-2 (from the lmms-lab team)
- microsoft/Phi-3-Vision (officially from the authors, Microsoft)
- LLaVA-SGlang (from the lmms-lab team)

## Citations

```shell
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
