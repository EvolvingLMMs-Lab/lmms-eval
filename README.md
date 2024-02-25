# lmms-eval

## How to run

```bash
pip install -e .
```

```bash
accelerate launch --num_processes=8 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-13b"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix debug --output_path ./logs/ # Eactly reproduce llava results
accelerate launch --num_processes=8 -m lmms_eval --config example_eval.yaml # Eactly reproduce llava results
```
## Current models

- GPT4V (API)
  - generation-based evaluation

- LLaVA-v1.5/v1.6-7B/13B/34B
  - generation-based evaluation
  - perplexity-based evaluation

- Qwen-VL
- Fuyu/OtterHD

## Models to be added

- InstructBLIP
- Emu
- CogVLM

## Current datasets
- AI2D (ai2d)
- ChartQA (chartqa)
- CMMMU (cmmmu)
  - CMMMU Validation (cmmmu_val)
  - CMMMU Test (cmmmu_test)
- COCO Caption (coco_cap)
  - COCO 2014 Caption (coco2014_cap)
    - COCO 2014 Caption Validation (coco2014_cap_val)
    - COCO 2014 Caption Test (coco2014_cap_test)
  - COCO 2017 Caption (coco2017_cap)
    - COCO 2017 Caption MiniVal (coco2017_cap_val)
    - COCO 2017 Caption MiniTest (coco2017_cap_test)
- DOCVQA (docvqa)
  - DOCVQA Validation (docvqa_val)
  - DOCVQA Test (docvqa_test)
- Ferret (ferret)
- Flickr30K (flickr30k)
  - Ferret Test (ferret_test)
- GQA (gqa)
- HallusionBenchmark (hallusion_bench_image)
- Infographic VQA (info_vqa)
  - Infographic VQA Validation (info_vqa_val)
  - Infographic VQA Test (info_vqa_test)
- Internal Eval (internal_eval)
  - D170 CN (d170_cn)
  - D170 EN (d170_en)
  - DC100 EN (dc100_en)
  - DC200 CN (dc200_cn)
- LLaVA-Bench (llava_bench_wild)
- LLaVA-Bench-COCO (llava_bench_coco)
- MathVista (mathvista)
  - MathVista Validation (mathvista_testmini)
  - MathVista Test (mathvista_test)
- MMBench (mmbench)
  - MMBench English (mmbench_en)
    - MMBench English Dev (mmbench_en_dev)
    - MMBench English Test (mmbench_en_test)
  - MMBench Chinese (mmbench_cn)
    - MMBench Chinese Dev (mmbench_cn_dev)
    - MMBench Chinese Test (mmbench_cn_test)
- MME (mme)
- MMMU (mmmu)
  - MMMU Validation (mmmu_val)
  - MMMU Test (mmmu_test)
- MMVet (mmvet)
- Multi-DocVQA (multidocvqa)
  - Multi-DocVQA Validation (multidocvqa_val)
  - Multi-DocVQA Test (multidocvqa_test)
- NoCaps (nocaps)
  - NoCaps Validation (nocaps_val)
  - NoCaps Test (nocaps_test)
- OKVQA (ok_vqa)
  - OKVQA Validation 2014 (ok_vqa_val2014)
- POPE (pope)
- RefCOCO (refcoco)
    - refcoco_seg_test
    - refcoco_seg_val
    - refcoco_seg_testA
    - refcoco_seg_testB
    - refcoco_bbox_test
    - refcoco_bbox_val
    - refcoco_bbox_testA
    - refcoco_bbox_testB
- RefCOCO+ (refcoco+)
    - refcoco+_seg
        - refcoco+_seg_val
        - refcoco+_seg_testA
        - refcoco+_seg_testB
    - refcoco+_bbox
        - refcoco+_bbox_val
        - refcoco+_bbox_testA
        - refcoco+_bbox_testB
- RefCOCOg (refcocog)
    - refcocog_seg_test
    - refcocog_seg_val
    - refcocog_bbox_test
    - refcocog_bbox_val
- ScienceQA (scienceqa_full)
  - ScienceQA Full (scienceqa)
  - ScienceQA IMG (scienceqa_img)
- SeedBench (seedbench)
- SeedBench 2 (seedbench_2)
- ST-VQA (stvqa)
- TextCaps (textcaps)
  - TextCaps Validation (textcaps_val)
  - TextCaps Test (textcaps_test)
- TextVQA (textvqa)
  - TextVQA Validation (textvqa_val)
  - TextVQA Test (textvqa_test)
- VizWizVQA (vizwiz_vqa)
  - VizWizVQA Validation (vizwiz_vqa_val)
  - VizWizVQA Test (vizwiz_vqa_test)
- VQAv2 (vqav2)
  - VQAv2 Validation (vqav2_val)
  - VQAv2 Test (vqav2_test)

## Datasets to be added and tested
- TallyQA (tallyqa)
- VSR (vsr)
- Winoground (winoground)
- NLVR2 (nlvr2)
- RavenIQ-Test (raveniq)
- IconQA (iconqa)
- VistBench (vistbench)


## Acknowledgement

The API, togegher with many code blocks of this project come from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). **Please read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) before contributing to this project**. Please do not commit to this project directly. Instead, push your changes to another branch and create a pull request.

Below are the changes we made to the original API:

- Instance.args (lmms_eval/api/instance.py) now contains a list of images to be inputted to lmms.
- lm-eval-harness supports all HF LMM as single model class. Currently this is not possible of lmms because the input/output format of lmms in HF are not yet unified. Thererfore, we have to create a new class for each lmms model. This is not ideal and we will try to unify them in the future.
