# Current Tasks

> () indicates the task name in lmms_eval. The task name is used to specify the dataset in the configuration file.

**Note:** This documentation is manually maintained. For the most up-to-date and complete list of supported tasks, please run:
```bash
python -m lmms_eval --tasks list
```

To see the number of questions in each task:
```bash
python -m lmms_eval --tasks list_with_num
```
(Note: `list_with_num` will download all datasets and may require significant time and storage)

---

## Summary Statistics

| Modality | Task Count |
|----------|------------|
| Image Understanding & VQA | 60+ |
| Multi-image Tasks | 15+ |
| Video Understanding | 25+ |
| Long Video & Temporal | 10+ |
| Audio & Speech | 20+ |
| Document Understanding | 12+ |
| Mathematical Reasoning | 12+ |
| Spatial & Grounding | 10+ |
| Text-only Language Tasks | 15+ |
| **Total** | **190+** |

---

## 1. Image Tasks

### Core VQA & Understanding Benchmarks
- [AI2D](https://arxiv.org/abs/1603.07396) (ai2d)
- [BLINK](https://github.com/jdf-prog/BLINK) (blink)
- [ChartQA](https://github.com/vis-nlp/ChartQA) (chartqa)
- [CharXiv](https://charxiv.github.io/) (charxiv)
- [COCO Caption](https://github.com/tylin/coco-caption) (coco_cap)
  - COCO 2014 Caption (coco2014_cap)
    - COCO 2014 Caption Validation (coco2014_cap_val)
    - COCO 2014 Caption Test (coco2014_cap_test)
  - COCO 2017 Caption (coco2017_cap)
    - COCO 2017 Caption MiniVal (coco2017_cap_val)
    - COCO 2017 Caption MiniTest (coco2017_cap_test)
- [ConBench](https://github.com/foundation-multimodal-models/ConBench) (conbench)
- [CV-Bench](https://github.com/nyu-visionx/CV-Bench) (cv_bench)
- [DetailCaps-4870](https://github.com/foundation-multimodal-models/CAPTURE) (detailcaps)
- [Flickr30K](https://github.com/BryanPlummer/flickr30k_entities) (flickr30k)
  - Flickr30K Test (flickr30k_test)
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html) (gqa)
- [GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru) (gqa_ru)
- [II-Bench](https://github.com/II-Bench/II-Bench) (ii_bench)
- [IllusionVQA](https://illusionvqa.github.io/) (illusionvqa)
- [LiveBench](https://huggingface.co/datasets/lmms-lab/LiveBench) (live_bench)
  - LiveBench 06/2024 (live_bench_2406)
  - LiveBench 07/2024 (live_bench_2407)
- [LLaVA-Bench-Wilder](https://huggingface.co/datasets/lmms-lab/LLaVA-Bench-Wilder) (llava_wilder_small)
- [LLaVA-Bench-COCO](https://llava-vl.github.io/) (llava_bench_coco)
- [LLaVA-Bench](https://llava-vl.github.io/) (llava_in_the_wild)
- [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) (naturalbench)
- [NoCaps](https://nocaps.org/) (nocaps)
  - NoCaps Validation (nocaps_val)
  - NoCaps Test (nocaps_test)
- [OKVQA](https://okvqa.allenai.org/) (ok_vqa)
  - OKVQA Validation 2014 (ok_vqa_val2014)
- [POPE](https://github.com/RUCAIBox/POPE) (pope)
- [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA) (realworldqa)
- [ScienceQA](https://scienceqa.github.io/) (scienceqa_full)
  - ScienceQA Full (scienceqa)
  - ScienceQA IMG (scienceqa_img)
- [SeedBench](https://github.com/AILab-CVC/SEED-Bench) (seedbench)
- [SeedBench 2](https://github.com/AILab-CVC/SEED-Bench) (seedbench_2)
- [SeedBench 2 Plus](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2-plus) (seedbench_2_plus)
- [VibeEval](https://github.com/reka-ai/reka-vibe-eval) (vibe_eval)
- [VizWizVQA](https://vizwiz.org/tasks-and-datasets/vqa/) (vizwiz_vqa)
  - VizWizVQA Validation (vizwiz_vqa_val)
  - VizWizVQA Test (vizwiz_vqa_test)
- [VQAv2](https://visualqa.org/) (vqav2)
  - VQAv2 Validation (vqav2_val)
  - VQAv2 Test (vqav2_test)
- [WildVision-Bench](https://github.com/WildVision-AI/WildVision-Bench) (wildvision)
  - WildVision 0617 (wildvision_0617)
  - WildVision 0630 (wildvision_0630)

### MMBench Family
- [MMBench](https://github.com/open-compass/MMBench) (mmbench)
  - MMBench English (mmbench_en)
    - MMBench English Dev (mmbench_en_dev)
    - MMBench English Test (mmbench_en_test)
  - MMBench Chinese (mmbench_cn)
    - MMBench Chinese Dev (mmbench_cn_dev)
    - MMBench Chinese Test (mmbench_cn_test)
- [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) (mme)
- [MME-CoT](https://huggingface.co/datasets/lmms-lab/MME-CoT) (mme_cot)
- [MME-RealWorld](https://mme-realworld.github.io/) (mmerealworld)
  - MME-RealWorld English (mmerealworld)
  - MME-RealWorld Mini (mmerealworld_lite)
  - MME-RealWorld Chinese (mmerealworld_cn)
- [MME-SCI](https://huggingface.co/datasets/JCruan/MME-SCI) (mme_sci)
  - MME-SCI (mme_sci)
  - MME-SCI-Image (mme_sci_images)
- [MMRefine](http://mmrefine.github.io/) (mmrefine)
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
- [MMVet v2](https://github.com/yuweihao/MM-Vet) (mmvetv2)
- [MMVU](https://mmvu-bench.github.io/) (mmvu)
- [MMWorld](https://mmworld-bench.github.io/) (mmworld)
- [MMSI-Bench](https://github.com/MMSI-Bench/MMSI-Bench) (mmsi_bench)
- [MMSearch](https://mmsearch.github.io/) (mmsearch)

### Hallucination & Bias Evaluation
- [HallusionBench](https://github.com/tianyi-lab/HallusionBench) (hallusion_bench_image)
- [VLMs Are Biased](https://github.com/vlms-are-biased/vlms-are-biased) (vlms_are_biased)
- [VLMs Are Blind](https://github.com/vlmsareblind/vlmsareblind) (vlmsareblind)

### Multilingual Benchmarks
- [Multilingual LLaVA Bench](https://huggingface.co/datasets/gagan3012/multilingual-llava-bench)
  - llava_in_the_wild_arabic
  - llava_in_the_wild_bengali
  - llava_in_the_wild_chinese
  - llava_in_the_wild_french
  - llava_in_the_wild_hindi
  - llava_in_the_wild_japanese
  - llava_in_the_wild_russian
  - llava_in_the_wild_spanish
  - llava_in_the_wild_urdu
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

### Quality & Low-level Vision
- [Q-Bench](https://q-future.github.io/Q-Bench/) (qbenchs_dev)
  - Q-Bench2-HF (qbench2_dev)
  - Q-Bench-HF (qbench_dev)
  - A-Bench-HF (abench_dev)
- [SalBench](https://salbench.github.io/) (salbench)
  - p3, p3_box, p3_box_img
  - o3, o3_box, o3_box_img
- [LV-Bench](https://github.com/LV-Bench/LV-Bench) (lvbench)
- [VSI-Bench](https://github.com/vsi-bench/vsi-bench) (vsibench)
- [HR-Bench](https://github.com/HR-Bench/HR-Bench) (hrbench)

### Specialized Vision Tasks
- [CameraBench VQA](https://github.com/CameraBench/CameraBench) (camerabench_vqa)
- [CUVA](https://github.com/CUVA-Lab/CUVA) (cuva)
- [DTC-Bench](https://github.com/DTC-Bench/DTC-Bench) (dtcbench)
- [FunQA](https://github.com/FunQA/FunQA) (funqa)
- [LiveXiv VQA](https://livexiv.github.io/) (livexiv_vqa)
- [LiveXiv TQA](https://livexiv.github.io/) (livexiv_tqa)
- [LSD-Bench](https://github.com/LSD-Bench/LSD-Bench) (lsdbench)
- [MIA-Bench](https://github.com/MIA-Bench/MIA-Bench) (mia_bench)
- [SNS-Bench](https://github.com/SNS-Bench/SNS-Bench) (snsbench)
- [TOMATO](https://github.com/TOMATO-Lab/TOMATO) (tomato)
- [VMC-Bench](https://github.com/VMC-Bench/VMC-Bench) (vmcbench)
- [Visual Puzzles](https://github.com/VisualPuzzles/VisualPuzzles) (VisualPuzzles)
- [VisualWebBench](https://visualwebbench.github.io/) (visualwebbench)
- [V*-Bench](https://github.com/V-Bench/V-Bench) (vstar_bench)
- [WorldSense](https://worldsense.github.io/) (worldsense)
- [AV-SpeakerBench](https://plnguyen2908.github.io/AV-SpeakerBench-project-page/) (av-speakerbench)

---

## 2. Multi-image Tasks

- [CMMMU](https://cmmmu-benchmark.github.io/) (cmmmu)
  - CMMMU Validation (cmmmu_val)
  - CMMMU Test (cmmmu_test)
- [HallusionBench](https://github.com/tianyi-lab/HallusionBench) (hallusion_bench_image)
- [ICON-QA](https://iconqa.github.io/) (iconqa)
  - ICON-QA Validation (iconqa_val)
  - ICON-QA Test (iconqa_test)
- [JMMMU](https://mmmu-japanese-benchmark.github.io/JMMMU/) (jmmmu)
- [JMMMU-Pro](https://mmmu-japanese-benchmark.github.io/JMMMU_Pro/) (jmmmu_pro)
- [LLaVA-NeXT-Interleave-Bench](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Interleave-Bench) (llava_interleave_bench)
  - llava_interleave_bench_in_domain
  - llava_interleave_bench_out_domain
  - llava_interleave_bench_multi_view
- [MIRB](https://github.com/ys-zong/MIRB) (mirb)
- [MMMU](https://mmmu-benchmark.github.io/) (mmmu)
  - MMMU Validation (mmmu_val)
  - MMMU Test (mmmu_test)
- [MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) (mmmu_pro)
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
- [OlympiadBench MIMO](https://github.com/OpenBMB/OlympiadBench) (olympiadbench_mimo)
- [MEGA-Bench](https://tiger-ai-lab.github.io/MEGA-Bench/) (megabench)
  - MEGA-Bench Core (megabench_core)
  - MEGA-Bench Open (megabench_open)
  - MEGA-Bench Core single-image subset (megabench_core_si)
  - MEGA-Bench Open single-image subset (megabench_open_si)

---

## 3. Video Tasks

### General Video Understanding
- [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) (activitynetqa_generation)
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
- [CinePile](https://cinepile.github.io/) (cinepile)
- [EgoSchema](https://github.com/egoschema/EgoSchema) (egoschema)
  - egoschema_mcppl
  - egoschema_subset_mcppl
  - egoschema_subset
- [EgoPlan](https://github.com/ChenYi99/EgoPlan) (egoplan)
- [EgoThink](https://github.com/AdaCheng/EgoThink) (egothink)
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
- [PerceptionTest](https://github.com/google-deepmind/perception_test) (perceptiontest)
  - PerceptionTest Test
    - perceptiontest_test_mc
    - perceptiontest_test_mcppl
  - PerceptionTest Validation
    - perceptiontest_val_mc
    - perceptiontest_val_mcppl
- [PLM VideoBench](https://github.com/PLM-VideoBench/PLM-VideoBench) (plm_videobench)
- [SciVideoBench](https://scivideobench.github.io/) (scivideobench)
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) (videochatgpt)
  - Video-ChatGPT Generic (videochatgpt_gen)
  - Video-ChatGPT Temporal (videochatgpt_temporal)
  - Video-ChatGPT Consistency (videochatgpt_consistency)
- [Video-MME](https://video-mme.github.io/) (videomme)
- [Video-MMMU](https://videommmu.github.io/) (videommmu)
- [VideoEval-Pro](https://tiger-ai-lab.github.io/VideoEval-Pro/) (videoevalpro)
- [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) (videomathqa)
- [Vinoground](https://vinoground.github.io) (vinoground)
- [WorldQA](https://zhangyuanhan-ai.github.io/WorldQA/) (worldqa)
  - WorldQA Generation (worldqa_gen)
  - WorldQA Multiple Choice (worldqa_mc)
- [YouCook2](http://youcook2.eecs.umich.edu/) (youcook2_val)

### Long Video & Temporal Understanding
- [Charades-STA](https://github.com/jiyanggao/TALL) (charades_sta)
- [FALCON-Bench](https://falcon-bench.github.io/) (FALCONBench) - One-hour-long video understanding
- [LEMONADE](https://huggingface.co/datasets/amathislab/LEMONADE) (lemonade)
- [LongTimescope](https://longtimescope.github.io/) (longtimescope)
- [LongVT](https://longvt-bench.github.io/) (longvt) - Tool-based long video understanding
- [LongVideoBench](https://github.com/longvideobench/LongVideoBench) (longvideobench)
- [MovieChat](https://github.com/rese1f/MovieChat) (moviechat)
  - Global Mode for entire video (moviechat_global)
  - Breakpoint Mode for specific moments (moviechat_breakpoint)
- [TempCompass](https://github.com/llyx97/TempCompass) (tempcompass)
  - tempcompass_multi_choice
  - tempcompass_yes_no
  - tempcompass_caption_matching
  - tempcompass_captioning
- [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) (temporalbench)
  - temporalbench_short_qa
  - temporalbench_long_qa
  - temporalbench_short_caption
- [Timescope](https://github.com/Timescope/Timescope) (timescope)

### Video Captioning & Description
- [Vatex](https://eric-xw.github.io/vatex-website/index.html) (vatex)
  - Vatex Chinese (vatex_val_zh)
  - Vatex Test (vatex_test)
- [VDC](https://github.com/rese1f/aurora) (vdc)
  - VDC Detailed Caption (detailed_test)
  - VDC Camera Caption (camera_test)
  - VDC Short Caption (short_test)
  - VDC Background Caption (background_test)
  - VDC Main Object Caption (main_object_test)
- [VideoDetailDescription](https://huggingface.co/datasets/lmms-lab/VideoDetailCaption) (video_dc499)
- [Video-TT](https://github.com/Video-TT/Video-TT) (video-tt)
- [VITATECS](https://github.com/lscpku/VITATECS) (vitatecs)
  - VITATECS Direction (vitatecs_direction)
  - VITATECS Intensity (vitatecs_intensity)
  - VITATECS Sequence (vitatecs_sequence)
  - VITATECS Compositionality (vitatecs_compositionality)
  - VITATECS Localization (vitatecs_localization)
  - VITATECS Type (vitatecs_type)

---

## 4. Audio & Speech Tasks

### Speech Recognition
- [Common Voice 15](https://commonvoice.mozilla.org/) (common_voice_15)
- [FLEURS](https://huggingface.co/datasets/google/fleurs) (fleurs)
- [GigaSpeech](https://github.com/SpeechColab/GigaSpeech) (gigaspeech)
- [LibriSpeech](https://www.openslr.org/12) (librispeech)
- [Open ASR](https://huggingface.co/datasets/esb/datasets) (open_asr)
- [People Speech](https://mlcommons.org/en/peoples-speech/) (people_speech)
- [TEDLium](https://www.openslr.org/51/) (tedlium)
- [WenetSpeech](https://wenet.org.cn/WenetSpeech/) (wenet_speech)
- [XLRS](https://huggingface.co/datasets/facebook/multilingual_librispeech) (xlrs)

### Speech Translation
- [CoVoST2](https://github.com/facebookresearch/covost) (covost2)

### Audio Understanding & QA
- [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) (air_bench)
  - air_bench_chat (chat-based audio understanding)
  - air_bench_foundation (foundation audio tasks)
- [Alpaca Audio](https://huggingface.co/datasets/alpaca-audio) (alpaca_audio)
- [AV-Odyssey](https://github.com/AV-Odyssey/AV-Odyssey) (av_odyssey)
- [Clotho-AQA](https://zenodo.org/record/6473207) (clotho_aqa)
- [MuchoMusic](https://huggingface.co/datasets/MuchoMusic/MuchoMusic) (muchomusic)
- [MMAU](https://github.com/MMAU-Bench/MMAU) (mmau)
- [Step2 Audio Paralinguistic](https://github.com/Step2-Audio/Step2-Audio) (step2_audio_paralinguistic)
- [VocalSound](https://github.com/YuanGongND/vocalsound) (vocalsound)
- [VoiceBench](https://voicebench.github.io/) (voicebench)
- [WavCaps](https://github.com/XinhaoMei/WavCaps) (wavcaps)

---

## 5. Document Understanding Tasks

- [DOCVQA](https://github.com/anisha2102/docvqa) (docvqa)
  - DOCVQA Validation (docvqa_val)
  - DOCVQA Test (docvqa_test)
- [GEdit-Bench](https://github.com/GEdit-Bench/GEdit-Bench) (gedit_bench)
- [Infographic VQA](https://www.docvqa.org/datasets/infographicvqa) (infovqa)
  - Infographic VQA Validation (infovqa_val)
  - Infographic VQA Test (infovqa_test)
- [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR) (ocrbench)
- [OCRBench v2](https://github.com/Yuliang-Liu/MultimodalOCR) (ocrbench_v2)
- [ScreenSpot](https://github.com/njucckevin/SeeClick) (screenspot)
  - ScreenSpot REC / Grounding (screenspot_rec)
  - ScreenSpot REG / Instruction Generation (screenspot_reg)
- [ST-VQA](https://rrc.cvc.uab.es/?ch=11) (stvqa)
- [SynthDog](https://github.com/clovaai/donut) (synthdog)
  - SynthDog English (synthdog_en)
  - SynthDog Chinese (synthdog_zh)
- [TextCaps](https://textvqa.org/textcaps/) (textcaps)
  - TextCaps Validation (textcaps_val)
  - TextCaps Test (textcaps_test)
- [TextVQA](https://textvqa.org/) (textvqa)
  - TextVQA Validation (textvqa_val)
  - TextVQA Test (textvqa_test)
- [WebSRC](https://x-lance.github.io/WebSRC/) (websrc)
  - WebSRC Validation (websrc_val)
  - WebSRC Test (websrc_test)

---

## 6. Mathematical Reasoning Tasks

- [AIME](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) (aime)
- [DynaMath](https://dynamath.github.io/) (dynamath)
- [GSM8K](https://github.com/openai/grade-school-math) (gsm8k)
- [MathVerse](https://github.com/ZrrSkywalker/MathVerse) (mathverse)
  - MathVerse Text Dominant (mathverse_testmini_text_dominant)
  - MathVerse Text Only (mathverse_testmini_text_only)
  - MathVerse Text Lite (mathverse_testmini_text_lite)
  - MathVerse Vision Dominant (mathverse_testmini_vision_dominant)
  - MathVerse Vision Intensive (mathverse_testmini_vision_intensive)
  - MathVerse Vision Only (mathverse_testmini_vision_only)
- [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) (mathvision)
  - MathVision TestMini (mathvision_testmini)
  - MathVision Test (mathvision_test)
  - MathVision Reason TestMini (mathvision_reason_testmini)
  - MathVision Reason Test (mathvision_reason_test)
- [MathVista](https://mathvista.github.io/) (mathvista)
  - MathVista Validation (mathvista_testmini)
  - MathVista Test (mathvista_test)
- [OpenAI Math](https://github.com/openai/prm800k) (openai_math)
- [SciBench](https://github.com/mandyyyyii/scibench) (scibench)
- [WeMath](https://wemath.github.io/) (wemath)

---

## 7. Spatial & Grounding Tasks

### Referring Expression Comprehension
- [Ferret](https://github.com/apple/ml-ferret) (ferret)
- [RefCOCO](https://github.com/lichengunc/refer) (refcoco)
  - refcoco_seg_test, refcoco_seg_val
  - refcoco_seg_testA, refcoco_seg_testB
  - refcoco_bbox_test, refcoco_bbox_val
  - refcoco_bbox_testA, refcoco_bbox_testB
- [RefCOCO+](https://github.com/lichengunc/refer) (refcoco+)
  - refcoco+_seg_val, refcoco+_seg_testA, refcoco+_seg_testB
  - refcoco+_bbox_val, refcoco+_bbox_testA, refcoco+_bbox_testB
- [RefCOCOg](https://github.com/lichengunc/refer) (refcocog)
  - refcocog_seg_test, refcocog_seg_val
  - refcocog_bbox_test, refcocog_bbox_val
- [RefSpatial](https://refspatial.github.io/) (refspatial)

### Spatial Reasoning
- [CS-Bench](https://csbench.github.io/) (csbench)
- [EmbSpatial](https://github.com/EmbSpatial/EmbSpatial) (embspatial)
- [ERQA](https://github.com/ERQA-Bench/ERQA) (erqa)
- [OmniSpatial](https://omnispatial.github.io/) (omnispatial)
- [Where2Place](https://where2place.github.io/) (where2place)

---

## 8. Text-only Language Tasks

- [ARC](https://allenai.org/data/arc) (arc)
- [GPQA](https://github.com/idavidrein/gpqa) (gpqa)
- [GSM8K](https://github.com/openai/grade-school-math) (gsm8k)
- [HellaSwag](https://rowanzellers.com/hellaswag/) (hellaswag)
- [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) (ifeval)
- [K12](https://github.com/K12-Benchmark/K12) (k12)
- [LogicVista](https://github.com/LogicVista/LogicVista) (logicvista)
- [MedQA](https://github.com/jind11/MedQA) (medqa)
- [MMLU](https://github.com/hendrycks/test) (mmlu)
- [MMLU_Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) (mmlu_pro)
- [OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5) (openhermes)
- [Super GPQA](https://github.com/SuperGPQA/SuperGPQA) (super_gpqa)

---

## 9. Multimodal Evaluation & Meta-benchmarks

- [Capability](https://github.com/Capability-Bench/Capability) (capability)
- [EMMA](https://github.com/EMMA-Bench/EMMA) (emma)
- [MindCube](https://mindcube.github.io/) (mindcube)
- [Mix Evals](https://github.com/mix-evals/mix-evals) (mix_evals)
- [Multimodal RewardBench](https://huggingface.co/datasets/allenai/reward-bench) (multimodal_rewardbench)
- [Omni-Bench](https://omni-bench.github.io/) (omni_bench)
- [PhyX](https://phyx-bench.github.io/) (phyx) - Physics grounded reasoning
- [UEval](https://github.com/UEval/UEval) (ueval)
- [VL-RewardBench](https://vl-rewardbench.github.io) (vl_rewardbench)

---

## 10. Supported Models

### Chat Template Models (Recommended)
| Model Name | Class | Description |
|------------|-------|-------------|
| `qwen2_5_vl` | Qwen2_5_VL | Qwen2.5-VL vision-language model |
| `qwen3_vl` | Qwen3_VL | Qwen3-VL vision-language model |
| `llava_hf` | LlavaHf | LLaVA via Hugging Face |
| `llava_onevision1_5` | Llava_OneVision1_5 | LLaVA-OneVision 1.5 |
| `thyme` | Thyme | Thyme multimodal model |
| `longvila` | LongVila | Long video understanding model |
| `bagel_lmms_engine` | BagelLmmsEngine | Bagel LMMS engine |
| `vllm` | VLLM | vLLM backend |
| `vllm_generate` | VLLMGenerate | vLLM generation mode |
| `sglang` | Sglang | SGLang serving backend |
| `huggingface` | Huggingface | Generic HuggingFace models |
| `openai_compatible` | OpenAICompatible | OpenAI-compatible APIs |
| `async_openai` | AsyncOpenAIChat | Async OpenAI chat |

### Simple/Legacy Models
| Model Name | Class | Modality |
|------------|-------|----------|
| `aero` | Aero | Image, Audio |
| `aria` | Aria | Image, Video |
| `auroracap` | AuroraCap | Image, Video captioning |
| `bagel` | Bagel | Image |
| `batch_gpt4` | BatchGPT4 | API |
| `claude` | Claude | Image, Video |
| `cogvlm2` | CogVLM2 | Image |
| `egogpt` | EgoGPT | Video |
| `from_log` | FromLog | Utility |
| `fuyu` | Fuyu | Image |
| `gemini_api` | GeminiAPI | Image, Audio |
| `gemma3` | Gemma3 | Image |
| `gpt4o_audio` | GPT4OAudio | Audio, Vision API |
| `gpt4v` | GPT4V | Image, Video API |
| `idefics2` | Idefics2 | Image |
| `instructblip` | InstructBLIP | Image |
| `internvideo2` | InternVideo2 | Video |
| `internvideo2_5` | InternVideo2_5 | Video |
| `internvl` | InternVLChat | Image |
| `internvl2` | InternVL2 | Image |
| `llama_vid` | LLaMAVid | Video |
| `llama_vision` | LlamaVision | Image |
| `llava` | Llava | Image |
| `llava_onevision` | Llava_OneVision | Image, Video |
| `llava_onevision_moviechat` | Llava_OneVision_MovieChat | Long Video |
| `llava_sglang` | LlavaSglang | Image (vLLM) |
| `llava_vid` | LlavaVid | Video |
| `longva` | LongVA | Long Video |
| `mantis` | Mantis | Multi-image |
| `minicpm_v` | MiniCPM_V | Image |
| `minimonkey` | MiniMonkey | Image |
| `moviechat` | MovieChat | Long Video |
| `mplug_owl_video` | mplug_Owl | Video |
| `ola` | Ola | Multimodal |
| `oryx` | Oryx | Multimodal |
| `phi3v` | Phi3v | Image |
| `phi4_multimodal` | Phi4 | Multimodal |
| `plm` | PerceptionLM | Image |
| `qwen_vl` | Qwen_VL | Image |
| `qwen_vl_api` | Qwen_VL_API | API |
| `qwen2_5_omni` | Qwen2_5_Omni | Image, Video, Audio |
| `qwen2_5_vl_interleave` | Qwen2_5_VL_Interleave | Interleaved |
| `qwen2_audio` | Qwen2_Audio | Audio |
| `qwen2_vl` | Qwen2_VL | Image, Video |
| `reka` | Reka | Multimodal API |
| `ross` | Ross | Multimodal |
| `slime` | Slime | Multimodal |
| `srt_api` | SRT_API | API |
| `tinyllava` | TinyLlava | Image |
| `videoChatGPT` | VideoChatGPT | Video |
| `video_llava` | VideoLLaVA | Video |
| `videochat2` | VideoChat2 | Video |
| `videochat_flash` | VideoChat_Flash | Video |
| `videollama3` | VideoLLaMA3 | Video |
| `vila` | VILA | Image, Video |
| `vita` | VITA | Multimodal |
| `vora` | VoRA | Multimodal |
| `whisper` | Whisper | Audio |
| `whisper_vllm` | WhisperVllm | Audio |
| `xcomposer2_4KHD` | XComposer2_4KHD | High-resolution Image |
| `xcomposer2d5` | XComposer2D5 | Image |

---

## Modality Support Summary

| Modality | Model Count |
|----------|-------------|
| Image | 50+ |
| Video | 20+ |
| Audio | 5+ |
| Multimodal (2+ modalities) | 15+ |
| API-based | 8+ |
| **Total Unique Models** | **70+** |
