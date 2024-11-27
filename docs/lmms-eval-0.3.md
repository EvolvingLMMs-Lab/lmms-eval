# Integration of Audio Evaluation in LMMs-Eval


# LMMs-Eval Analysis - Release of Audio Evaluations

## **Introduction**

Humans perceive the world through both sight and sound, integrating visual cues with auditory signals such as speech, environmental sounds, and emotional tones. 

This dual sensory input enhances decision-making and overall understanding. Similarly, for multimodal models to achieve human-like comprehension, it is essential to make them process both visual and auditory data together.

While many models have made progress in integrating audio understanding, there is still no reproducible and efficient evaluation toolkit to fairly assess their capabilities.

To address this, we introduce an upgrade to the `lmms-eval` framework, focusing on audio understanding. Building on the success of `lmms-eval/v0.2.0`, the new `lmms-eval/v0.3.0` includes dedicated modules and designs for audio tasks, ensuring consistent evaluation across audio and visual modalities. 

This upgrade includes multiple benchmarks for audio understanding and instruction following, enabling standardized and reproducible comparisons of various audio models.

## Audio Evaluation Pipeline

1. **Improved Pipeline for Audio Evaluations**
    
    Here’s a breakdown of adding audio datasets support.
    
    1. **Load Audio:** Audios are saved in HuggingFace and can be loaded via the  `doc_to_audio` function.
        - The code specifically demonstrates the logic of how we handle audio datasets in lmms-eval.
            
            ```python
            def air_bench_doc_to_audio(doc):
                return [doc["audio"]]
            ```
            
    2. **Format questions:**  Questions and instructions are defined in `<taskname>/utils.py`. For some Audio Instruction Following (AIF) tasks, we create custom prompts and try to align with Qwen2-Audio's evaluation format since the default dataset instructions are sometimes not clear enough for some datasets. We can add model-specific prompts besides the default instruction.
        - The code demonstrates an example of formatting the question.
            
            ```python
            # This is the place where you format your question
            def common_voice_15_doc_to_text(doc, lmms_eval_specific_kwargs):
                pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
                post_prompt = lmms_eval_specific_kwargs["post_prompt"]
                return f"{pre_prompt}Please recognize the speech and only output the recognized content:{post_prompt}"
            ```
            
    3. **Process results:**  Model outputs are evaluated using metrics from either official dataset implementations or aligning with the implementation in [AudioBench](https://github.com/AudioLLMs/AudioBench). We primarily adopt three types of metrics:
        
        **a. Accuracy:** Used for tasks with definitive ground truth answers, such as multiple-choice questions

        **b. WER:** Applied to some Audio Speech Recognition (ASR) tasks.

        **c. GPT-4 Eval:** Applied to open-ended responses. We align the evaluation prompt with the implementation in [AudioBench](https://github.com/AudioLLMs/AudioBench).
        
        - The code specifically demonstrates an example prompt for GPT-4 Evaluation.
            
            ```python
            eval_prompt = """
                        [Question]
                        {question}
            
                        [Reference Answer]
                        {ground_truth}
            
                        [Model Answer]
                        {model_response}
            
                        [Task]
                        Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
                        Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
                        Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
                        Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
                        Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
                        Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
                        Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
                        Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.
            
                        Your response should be formatted as follows:
                        Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
                        Rating: (int)"""
            ```
            
        

        
    4. **Aggregate results:** 
    After evaluating each data instance, we aggregate the individual results to generate the overall evaluation metrics. Finally, we provide a summary table that consolidates all the evaluation results, similar to the one in [Google’s Gemini report](https://arxiv.org/abs/2312.11805).
    5. **Grouped Tasks:** 
    For tasks with multiple subsets, we group all subset tasks together. For example, the AirBench-Chat dataset includes 4 subsets: sound, music, speech, mixed. By running `--task air_bench_chat`, all 4 subsets can be evaluated together, eliminating the need to specify each subset individually. We summarize all the grouped task names in Table 1. This pipeline ensures a thorough and standardized evaluation process for Audio, facilitating consistent and reliable performance assessment across various tasks and datasets.
        - The code specifically demonstrates an example yaml file of task grouping.
            
            ```python
            group: air_bench_chat
            task:
            - air_bench_chat_sound
            - air_bench_chat_music
            - air_bench_chat_speech
            - air_bench_chat_mixed
            ```
        
    
2. **Audio-based Capabilities**

    Our selected benchmarks assess the following key audio processing abilities, as inspired by [AudioBench](https://github.com/AudioLLMs/AudioBench):

    1. **Audio Captioning:** The ability to accurately transcribe human speech and convert audio content into text
    2. **Speech Understanding:** The capability to comprehend the semantic meaning of human speech, enabling appropriate responses to questions and audio instructions
    3. **Audio Scene Understanding:** The ability to interpret non-human sounds, such as environment sounds
    4. **Voice Understanding:** The capability to analyze non-speech human vocal information, including emotional states, accents, and speaker characteristics
    5. **Specialized Audio Processing:** The ability to analyze other audio types, such as musical compositions and multilingual content

    Our selected audio benchmarks collectively form a comprehensive evaluation of different audio-based capabilities across diverse scenarios.

### **Meta Information for Audio Datasets**

##### Table 1: Meta informantion for audio datasets

| **Dataset** | **Year** | **Task Name in lmms-eval** | **Split** | **Task Format** | **Evaluation Metric** | **Number of QAs** | **Feature** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **AIRBench** | 2024 | air_bench_chat \| air_bench_foundation | chat, foundation | AIF | GPT-4 Eval (chat) \| Accuracy (foundation) | 2k (chat) \| 19k (foundation) | Comprhensive tasks and audio types |
| **Alpaca Audio** | 2024 | alpaca_audio | test | AIF | GPT-4 Eval | 100 | synthetic voice |
| **Clotho-AQA** | 2022 | clotho_aqa | test \| val | AIF | Accuracy | test_v2 (2.06k), test \| val (1.44k \| 1.05k) | 1. Audio Question Answering<br> 2. single word answer<br> 3. text based question |
| **Common_voice** | 2023 | common_voice_15 | test | ASR | WER (align with Qwen-audio) | en (16.4k) \| fr (16.1k) \| zh (10.6k) | 1. real people voice<br> 2. captioning |
| **GigaSpeech** | 2021 | gigaspeech | test \| dev | ASR | WER | dev (6.75k) \| test (25.6k) | 1. transciption<br> 2. audio book<br> 3. YouTube<br> 4. podcasts |
| **LibriSpeech** | 2015 | librispeech | dev-clean \| dev-other \| test-clean \| test-other | ASR | WER | dev-clean (~2.48k) \|dev-other (~2.66k) \|test-clean(~2.55k) \| test-other (~2.70k) | Transcription (audio book) |
| **OpenHermes** | 2024 | openhermes | test | AIF | GPT-Eval | 100 | synthetic voice |
| **MuchoMusic** | 2024 | muchomusic | test | AIF | Accuracy | 1.19k | Music understanding |
| **People_speech** | 2021 | people_speech_val | val | ASR | WER | 18.6k | 1. real people voice<br> 2. captioning |
| **Tedium v3** | 2018 | tedlium_dev_test | val | ASR | WER | 591 | 1. ted talk<br>  2. real people asr<br>  3. captioning |
| **VocalSound** | 2022 | vocalsound_test | test \| val | AIF | Accuracy | test (3.59k) | val (1.86k) | 1. Vocal sound recognition<br> 2. Non-speech |
| **WavCaps** | 2024 | wavcaps | test | ASR | GPT-4 Eval | 1.73k | 1. Audio Captioning<br> 2. ChatGPT-augmented captions |

### Alignment Check for Audio Datasets

##### Table 2: Alignment check for audio datasets

|  |  | **metric** | **Qwen2-Audio-Instruct (lmms-eval)** | **Qwen2-Audio (lmms-eval)** |
| --- | --- | --- | --- | --- |
| **AIRBench-Chat** | Speech | GPT-Eval | 7.16 |  |
|  | Sound |  | 6.14 |  |
|  | Music |  | 6.66 |  |
|  | Mixed |  | 5.75 |  |
| **AIRBench-Foundation** | Speech | Acc | 62.89 |  |
|  | Sound |  | 55.42 |  |
|  | Music |  | 56.77 |  |
| **Alpaca** | test | GPT-Eval | 51.8 |  |
| **Clotho_aqa** | test | GPT-Eval | 0.7587 |  |
| **Common_voice** | zh |WER| 15.78 | 6.7 |
|  | en |  | 36.01 | 27.9 |
|  | fr |  | 39.88 | 34.8 |
| **GigaSpeech** | dev |WER| 19.45 | 14 |
|  | test |  | 22.6 | 15.01 |
| **LibriSpeech** | dev-clean |WER| 4.24 | 1.66 |
|  | dev-others |  | 6.54 | 3.66 |
|  | test-clean |  | 3.59 | 1.74 |
|  | test-others |  | 7.46 | 3.87 |
| **MuchoMusic** | test | Acc | 68.32 | 45.07 |
| **OpenHermes** | test | GPT-Eval | 46.8 |  |
| **People_speech** | val |WER| 25.86 | 17.1 |
| **Tedium** | val |WER| 10.92 | 8.29 |
| **VocalSound** | test | Acc | 0.936 | 0.81 |
|  | val |  | 0.9288 | 0.8 |
| **WavCaps** | test | GPT-Eval | 1.73 |  |


    The result might be inconsistent with the reported result as we do not have the original prompt and we have to maintain the fair environment for all the models. For the base model, we do not test on the Chat Benchmarks.

    Certain datasets face alignment challenge: Datasets with WER, CIDEr, BLEU as metrics cannot accurately align due to their rigid, reference-based formats. Model response sensitive to prompt, we will investigate more deeply in Section [Robustness of the model](https://www.notion.so/Robustness-of-the-model-b89c005d3e044cb6aff51165929cea45?pvs=21) .

## Evaluation Analysis and Thinking:

    During our implementation, we observe several interesting phenomena that may be valuable to discuss. We believe that reflecting on these aspects deeply can help accelerate the development of truly robust audio evaluations.

### Robustness of the model

    As we trying to align the results, our investigation revealed that the choice of chat template significantly impacts model performance, even for instruction-tuned models. This finding emerged while analyzing the Qwen2 Audio model. The original Qwen2 Audio repository uses a minimal prompt format:  `"<|audio_bos|><|AUDIO|><|audio_eos|>"` .

    This basic format is then combined with various question prompts for different evaluation scenarios. However, this prompt format is not in an instruction format and when applying a chat template, the performance of the model may changes significantly.

##### Table 3: Impact of Chat Template on Qwen-7B-Instruct's Performance

| Impact of Chat Template |  |  | Chat Template (Off) | Chat Template (On) |
| --- | --- | --- | --- | --- |
| LibriSpeech | dev-clean | WER(↓) | 2.65 | 4.24 |
|  | dev-others |  | 5.36 | 6.54 |
|  | test-clean |  | 2.91 | 3.59 |
|  | test-others |  | 5.14 | 7.46 |
| People_speech | val | WER(↓) | 21.92 | 25.86 |
| Tedium | dev_test | WER(↓) | 9.56 | 10.92 |

    More specifically, we founds out that as shown in the above table, the influence of the chat template is very huge. We believe that these demonstrate the actual robustness of the model and signifies that current audio model may eventually not being stable enough when coping different text input. Also, it again leads us into another thinking: “Is current metrics good at evaluating a model’s performance?

### Rethinking the evaluation metrics

    Traditional fixed-format metrics like WER, CIDEr, and BLEU face several limitations in audio model evaluation:

    1. **Format Rigidity:** Fixed metrics struggle to properly assess responses that are semantically correct but differ in format from reference answers
    2. **Prompt Sensitivity:** These metrics are highly sensitive to variations in input prompts, leading to inconsistent evaluation results

    Due to these limitations, the scores reported in `lmms-eval` might slightly differ from those reported in original papers, highlighting the challenge of maintaining consistent evaluation standards across different frameworks.

    Looking ahead, model-based evaluators such as GPT-4 could offer a more flexible and robust evaluation approach. Such evaluators can better understand semantic meaning, handle diverse response formats, and provide more consistent scoring across different implementations. This shift from rigid metrics to intelligent evaluation systems may better capture the true capabilities of audio processing models.

## Additional Experiments

### Batch Size

    We perform an exploratory batch inference experiment on Qwen2-Audio with the following results:

##### Table 4: Impact of batch size

|  | **Split** | **Metric** | **Qwen2-Audio (BS=4)** | **Qwen2-Audio (BS=1)** |
| --- | --- | --- | --- | --- |
| **LibriSpeech** | dev-clean | wer(↓) | 1.66 | 1.66 |
|  | dev-others |  | 4.4 | 3.66 |
|  | test-clean |  | 1.75 | 1.74 |
|  | test-others |  | 4.06 | 3.87 |
| **Total Time** |  |  | 10 mins 50 seconds | 5 min 23 seconds |

    As shown in the above results, the batch inference (BS=4) can significantly saves the inference time, it could lead to evaluation inconsistencies compared to single-sample processing (BS=1). This is a known issue in the `transformers` library that currently lacks a solution.

### More Details and Feature Updates with `v0.3.0`

1. **Supported Audio Tasks**
    1. [AirBench](https://github.com/OFA-Sys/AIR-Bench)
    2. [Alpaca Audio](https://tango2-web.github.io/)
    3. [Clotho-AQA](https://github.com/partha2409/AquaNet)
    4. [Common_voice_15](https://github.com/common-voice/common-voice)
    5. [GigaSpeech](https://github.com/SpeechColab/GigaSpeech)
    6. [LibriSpeech](https://www.openslr.org/12)
    7. [OpenHermes](https://huggingface.co/datasets/AudioLLMs/openhermes_instruction_test)
    8. [MuchoMusic](https://github.com/mulab-mir/muchomusic)
    9. [Peoples_speech](https://mlcommons.org/datasets/peoples-speech/)
    10. [Tedium v3](https://www.openslr.org/51/)
    11. [VocalSound](https://github.com/YuanGongND/vocalsound)
    12. [WavCaps](https://github.com/XinhaoMei/WavCaps)
2. **Support Audio Models**
    
    1. [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
    2. [Gemini_Audio](https://arxiv.org/abs/2312.11805)
3. **Supporting Multi-Round Evaluation**
    1. [Feat][Task] Add multi-round evaluation in llava-onevision; Add MMSearch Benchmark by [@CaraJ7](https://github.com/CaraJ7) in [#277](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/277)
4. **Regression Test**
    1. [Feat] add regression test and change saving logic related to `output_path` by [@Luodian](https://github.com/Luodian) in [#259](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/259)
5. **Speed-Up by loading required tasks and models.**
    1. [feat] remove registeration logic and adding language evaluation tasks. by [@Luodian](https://github.com/Luodian) in [#218](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/218)
6. **LMMs-Eval Analysis Tool**
    1. Lite/Core-set Selection by Kaichen Zhang
        
        https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/tools/lite
        
    2. LiveBench by Fanyi Pu
        
        https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/tools/live_bench
        
7. **SGLang Evaluation**
    1. [Feat] SGLang SRT commands in one go, async input for openai server by [@kcz358](https://github.com/kcz358) in [#212](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/212)
    2. [Fix] Fix async append result in different order issue by [@kcz358](https://github.com/kcz358) in [#244](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/244)

## Contributors

> Listed in order of contribution significance.
> 

**Core Contributors**

Pengyun Wang, Cong Pham Ba, Yingluo Li, Fanyi Pu

**Release Managers**

Kairui Hu, Kaichen Zhang, Bo Li
