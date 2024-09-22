# Current Tasks

> () indicates the task name in the lmms_eval. The task name is also used to specify the dataset in the configuration file.
> The following is manually updated documentation. You could use `lmms_eval task --list` to list all supported tasks and their task names.

## 1. Image tasks:

- [AI2D](https://arxiv.org/abs/1603.07396) (ai2d)
- [ChartQA](https://github.com/vis-nlp/ChartQA) (chartqa)
- [COCO Caption](https://github.com/tylin/coco-caption) (coco_cap)
  - COCO 2014 Caption (coco2014_cap)
    - COCO 2014 Caption Validation (coco2014_cap_val)
    - COCO 2014 Caption Test (coco2014_cap_test)
  - COCO 2017 Caption (coco2017_cap)
    - COCO 2017 Caption MiniVal (coco2017_cap_val)
    - COCO 2017 Caption MiniTest (coco2017_cap_test)
- [ConBench](https://github.com/foundation-multimodal-models/ConBench) (conbench)
- [DetailCaps-4870](https://github.com/foundation-multimodal-models/CAPTURE) (detailcaps)
- [DOCVQA](https://github.com/anisha2102/docvqa) (docvqa)
  - DOCVQA Validation (docvqa_val)
  - DOCVQA Test (docvqa_test)
- [Ferret](https://github.com/apple/ml-ferret) (ferret)
- [Flickr30K](https://github.com/BryanPlummer/flickr30k_entities) (flickr30k)
  - Flickr30K Test (flickr30k_test)
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html) (gqa)
- [GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru) (gqa_ru)
- [II-Bench](https://github.com/II-Bench/II-Bench) (ii_bench)
- [Infographic VQA](https://www.docvqa.org/datasets/infographicvqa) (infovqa)
  - Infographic VQA Validation (infovqa_val)
  - Infographic VQA Test (infovqa_test)
- [LiveBench](https://huggingface.co/datasets/lmms-lab/LiveBench) (live_bench)
  - LiveBench 06/2024 (live_bench_2406)
  - LiveBench 07/2024 (live_bench_2407)
- [LLaVA-Bench-Wilder](https://huggingface.co/datasets/lmms-lab/LLaVA-Bench-Wilder) (llava_wilder_small)
- [LLaVA-Bench-COCO](https://llava-vl.github.io/) (llava_bench_coco)
- [LLaVA-Bench](https://llava-vl.github.io/) (llava_in_the_wild)
- [MathVerse](https://github.com/ZrrSkywalker/MathVerse) (mathverse)
  - MathVerse Text Dominant (mathverse_testmini_text_dominant)
  - MathVerse Text Only (mathverse_testmini_text_only)
  - MathVerse Text Lite (mathverse_testmini_text_lite)
  - MathVerse Vision Dominant (mathverse_testmini_vision_dominant)
  - MathVerse Vision Intensive (mathverse_testmini_vision_intensive)
  - MathVerse Vision Only (mathverse_testmini_vision_only)
- [MathVista](https://mathvista.github.io/) (mathvista)
  - MathVista Validation (mathvista_testmini)
  - MathVista Test (mathvista_test)
- [MMBench](https://github.com/open-compass/MMBench) (mmbench)
  - MMBench English (mmbench_en)
    - MMBench English Dev (mmbench_en_dev)
    - MMBench English Test (mmbench_en_test)
  - MMBench Chinese (mmbench_cn)
    - MMBench Chinese Dev (mmbench_cn_dev)
    - MMBench Chinese Test (mmbench_cn_test)
- [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) (mme)
- [MMStar](https://github.com/MMStar-Benchmark/MMStar) (mmstar)
- [MMUPD](https://huggingface.co/datasets/MM-UPD/MM-UPD) (mmupd)
  - MMUPD Base (mmupd_base)
    - MMAAD Base (mmaad_base)
    - MMIASD Base (mmiasd_base)
    - MMIVQD Base (mmivqd_base)
  - MMUPD Option (mmupd_option)
    - MMAAD Option (mmaad_option)
    - MMIASD Option (mmiasd_option)
    - MMIVQD Option (mmivqd_option)
  - MMUPD Instruction (mmupd_instruction)
    - MMAAD Instruction (mmaad_instruction)
    - MMIASD Instruction (mmiasd_instruction)
    - MMIVQD Instruction (mmivqd_instruction)
- [MMVet](https://github.com/yuweihao/MM-Vet) (mmvet)
- [Multilingual LlaVa Bench](https://huggingface.co/datasets/gagan3012/multilingual-llava-bench)
  - llava_in_the_wild_arabic
  - llava_in_the_wild_bengali
  - llava_in_the_wild_chinese
  - llava_in_the_wild_french
  - llava_in_the_wild_hindi
  - llava_in_the_wild_japanese
  - llava_in_the_wild_russian
  - llava_in_the_wild_spanish
  - llava_in_the_wild_urdu
- [NoCaps](https://nocaps.org/) (nocaps)
  - NoCaps Validation (nocaps_val)
  - NoCaps Test (nocaps_test)
- [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR) (ocrbench)
- [OKVQA](https://okvqa.allenai.org/) (ok_vqa)
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
  - refcoco+\_seg
    - refcoco+\_seg_val
    - refcoco+\_seg_testA
    - refcoco+\_seg_testB
  - refcoco+\_bbox
    - refcoco+\_bbox_val
    - refcoco+\_bbox_testA
    - refcoco+\_bbox_testB
- RefCOCOg (refcocog)
  - refcocog_seg_test
  - refcocog_seg_val
  - refcocog_bbox_test
  - refcocog_bbox_val
- ScienceQA (scienceqa_full)
  - ScienceQA Full (scienceqa)
  - ScienceQA IMG (scienceqa_img)
- [ScreenSpot](https://github.com/njucckevin/SeeClick) (screenspot)
  - ScreenSpot REC / Grounding (screenspot_rec)
  - ScreenSpot REG / Instruction Generation (screenspot_reg)
- [ST-VQA](https://rrc.cvc.uab.es/?ch=11) (stvqa)
- [synthdog](https://github.com/clovaai/donut) (synthdog)
  - synthdog English (synthdog_en)
  - synthdog Chinese (synthdog_zh)
- [TextCaps](https://textvqa.org/textcaps/) (textcaps)
  - TextCaps Validation (textcaps_val)
  - TextCaps Test (textcaps_test)
- [TextVQA](https://textvqa.org/) (textvqa)
  - TextVQA Validation (textvqa_val)
  - TextVQA Test (textvqa_test)
- [VCR-Wiki](https://github.com/tianyu-z/VCR)
  - VCR-Wiki English
    - VCR-Wiki English easy 100 (vcr_wiki_en_easy_100)
    - VCR-Wiki English easy 500 (vcr_wiki_en_easy_500)
    - VCR-Wiki English easy (vcr_wiki_en_easy)
    - VCR-Wiki English hard 100 (vcr_wiki_en_hard_100)
    - VCR-Wiki English hard 500 (vcr_wiki_en_hard_500)
    - VCR-Wiki English hard (vcr_wiki_en_hard)
  - VCR-Wiki Chinese
    - VCR-Wiki Chinese easy 100 (vcr_wiki_zh_easy_100)
    - VCR-Wiki Chinese easy 500 (vcr_wiki_zh_easy_500)
    - VCR-Wiki Chinese easy (vcr_wiki_zh_easy)
    - VCR-Wiki Chinese hard 100 (vcr_wiki_zh_hard_100)
    - VCR-Wiki Chinese hard 500 (vcr_wiki_zh_hard_500)
    - VCR-Wiki Chinese hard (vcr_wiki_zh_hard)
- [VibeEval](https://github.com/reka-ai/reka-vibe-eval) (vibe_eval)
- [VizWizVQA](https://vizwiz.org/tasks-and-datasets/vqa/) (vizwiz_vqa)
  - VizWizVQA Validation (vizwiz_vqa_val)
  - VizWizVQA Test (vizwiz_vqa_test)
- [VQAv2](https://visualqa.org/) (vqav2)
  - VQAv2 Validation (vqav2_val)
  - VQAv2 Test (vqav2_test)
- [WebSRC](https://x-lance.github.io/WebSRC/) (websrc)
  - WebSRC Validation (websrc_val)
  - WebSRC Test (websrc_test)
- [WildVision-Bench](https://github.com/WildVision-AI/WildVision-Bench) (wildvision)
  - WildVision 0617(wildvision_0617)
  - WildVision 0630 (wildvision_0630)
- [SeedBench 2 Plus](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2-plus) (seedbench_2_plus)

## 2. Multi-image tasks:

- [CMMMU](https://cmmmu-benchmark.github.io/) (cmmmu)
  - CMMMU Validation (cmmmu_val)
  - CMMMU Test (cmmmu_test)
- [HallusionBench](https://github.com/tianyi-lab/HallusionBench) (hallusion_bench_image)
- [ICON-QA](https://iconqa.github.io/) (iconqa)
  - ICON-QA Validation (iconqa_val)
  - ICON-QA Test (iconqa_test)
- [LLaVA-NeXT-Interleave-Bench](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Interleave-Bench) (llava_interleave_bench)
  - llava_interleave_bench_in_domain
  - llava_interleave_bench_out_domain
  - llava_interleave_bench_multi_view
- [MIRB](https://github.com/ys-zong/MIRB) (mirb)
- [MMMU](https://mmmu-benchmark.github.io/) (mmmu)
  - MMMU Validation (mmmu_val)
  - MMMU Test (mmmu_test)
- [MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro)
  - MMMU Pro (mmmu_pro)
    - MMMU Pro Original (mmmu_pro_original)
    - MMMU Pro Vision (mmmu_pro_vision)
  - MMMU Pro COT (mmmu_pro_cot)
    - MMMU Pro Original COT (mmmu_pro_original_cot)
    - MMMU Pro Vision COT (mmmu_pro_vision_cot)
    - MMMU Pro Composite COT (mmmu_pro_composite_cot)
- [MMT Multiple Image](https://mmt-bench.github.io/) (mmt_mi)
  - MMT Multiple Image Validation (mmt_mi_val)
  - MMT Multiple Image Test (mmt_mi_test)
- [MuirBench](https://muirbench.github.io/) (muirbench)
- [MP-DocVQA](https://github.com/rubenpt91/MP-DocVQA-Framework) (multidocvqa)
  - MP-DocVQA Validation (multidocvqa_val)
  - MP-DocVQA Test (multidocvqa_test)
- [OlympiadBench](https://github.com/OpenBMB/OlympiadBench) (olympiadbench)
  - OlympiadBench Test English (olympiadbench_test_en)
  - OlympiadBench Test Chinese (olympiadbench_test_cn)
- [Q-Bench](https://q-future.github.io/Q-Bench/) (qbenchs_dev)
  - Q-Bench2-HF (qbench2_dev)
  - Q-Bench-HF (qbench_dev)
  - A-Bench-HF (abench_dev)

## 3. Videos tasks:

- [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) (activitynetqa_generation)
- [SeedBench](https://github.com/AILab-CVC/SEED-Bench) (seedbench)
- [SeedBench 2](https://github.com/AILab-CVC/SEED-Bench) (seedbench_2)
- [CVRR-ES](https://github.com/mbzuai-oryx/CVRR-Evaluation-Suite) (cvrr)
  - cvrr_continuity_and_object_instance_count
  - cvrr_fine_grained_action_understanding
  - cvrr_interpretation_of_social_context
  - cvrr_interpretation_of_visual_context
  - cvrr_multiple_actions_in_a_single_video
  - cvrr_non_existent_actions_with_existent_scene_depictions
  - cvrr_non_existent_actions_with_non_existent_scene_depictions
  - cvrr_partial_actions
  - cvrr_time_order_understanding
  - cvrr_understanding_emotional_context
  - cvrr_unusual_and_physically_anomalous_activities
- [EgoSchema](https://github.com/egoschema/EgoSchema) (egoschema)
  - egoschema_mcppl
  - egoschema_subset_mcppl
  - egoschema_subset
- [LongVideoBench](https://github.com/longvideobench/LongVideoBench)
- [MLVU](https://github.com/JUNJIE99/MLVU) (mlvu)
- [MMT-Bench](https://mmt-bench.github.io/) (mmt)
  - MMT Validation (mmt_val)
  - MMT Test (mmt_test)
- [MVBench](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) (mvbench)

  - mvbench_action_sequence
  - mvbench_moving_count
  - mvbench_action_prediction
  - mvbench_episodic_reasoning
  - mvbench_action_antonym
  - mvbench_action_count
  - mvbench_scene_transition
  - mvbench_object_shuffle
  - mvbench_object_existence
  - mvbench_fine_grained_pose
  - mvbench_unexpected_action
  - mvbench_moving_direction
  - mvbench_state_change
  - mvbench_object_interaction
  - mvbench_character_order
  - mvbench_action_localization
  - mvbench_counterfactual_inference
  - mvbench_fine_grained_action
  - mvbench_moving_attribute
  - mvbench_egocentric_navigation

- [NExT-QA](https://github.com/doc-doc/NExT-QA) (nextqa)

  - NExT-QA Multiple Choice Test (nextqa_mc_test)
  - NExT-QA Open Ended Validation (nextqa_oe_val)
  - NExT-QA Open Ended Test (nextqa_oe_test)

- [PerceptionTest](https://github.com/google-deepmind/perception_test)

  - PerceptionTest Test
    - perceptiontest_test_mc
    - perceptiontest_test_mcppl
  - PerceptionTest Validation
    - perceptiontest_val_mc
    - perceptiontest_val_mcppl

- [TempCompass](https://github.com/llyx97/TempCompass) (tempcompass)

  - tempcompass_multi_choice
  - tempcompass_yes_no
  - tempcompass_caption_matching
  - tempcompass_captioning

- [Vatex](https://eric-xw.github.io/vatex-website/index.html) (vatex)

  - Vatex Chinese (vatex_val_zh)
  - Vatex Test (vatex_test)

- [VideoDetailDescription](https://huggingface.co/datasets/lmms-lab/VideoDetailCaption) (video_dc499)
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) (videochatgpt)
  - Video-ChatGPT Generic (videochatgpt_gen)
  - Video-ChatGPT Temporal (videochatgpt_temporal)
  - Video-ChatGPT Consistency (videochatgpt_consistency)
- [Video-MME](https://video-mme.github.io/) (videomme)
- [VITATECS](https://github.com/lscpku/VITATECS) (vitatecs)

  - VITATECS Direction (vitatecs_direction)
  - VITATECS Intensity (vitatecs_intensity)
  - VITATECS Sequence (vitatecs_sequence)
  - VITATECS Compositionality (vitatecs_compositionality)
  - VITATECS Localization (vitatecs_localization)
  - VITATECS Type (vitatecs_type)

- [WorldQA](https://zhangyuanhan-ai.github.io/WorldQA/) (worldqa)

  - WorldQA Generation (worldqa_gen)
  - WorldQA Multiple Choice (worldqa_mc)

- [YouCook2](http://youcook2.eecs.umich.edu/) (youcook2_val)

## 4. Text Tasks

- [GSM8K](https://github.com/openai/grade-school-math) (gsm8k)
- [HellaSwag](https://rowanzellers.com/hellaswag/) (hellaswag)
- [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) (ifeval)
- [MMLU](https://github.com/hendrycks/test) (mmlu)
- [MMLU_pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) (mmlu_pro)
