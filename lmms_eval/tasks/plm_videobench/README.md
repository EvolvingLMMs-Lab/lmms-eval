# PLM-VideoBench
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#8209;VideoBench-BenchMark-blue)](https://huggingface.co/datasets/facebook/PLM-VideoBench)
[![Paper](https://img.shields.io/badge/Technical%20Report-PerceptionLM-b31b1b.svg)](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding)

PLM-VideoBench is a comprehensive set of video benchmarks for detailed video understanding. It includes the following sub-benchmarks,
1. **Fine-Grained Question Answering (FGQA):** In this task, a model must answer a multiple-choice question (MCQ)
that probes fine-grained activity understanding.
2. **Smart Glasses Question Answering (SGQA):** In this task, a model must answer open-ended questions about
activities and objects visible in an egocentric video stream recorded by a Meta VR Glasses.
3. **Video Region Captioning (RCap):** In this task, the model must generate a detailed description of an event
involving a subject of interest in the video. 
4. **Region Temporal Localization (RTLoc):** In this task, the model must identify the precise time interval within the video when the specified event takes place for the given subject.
5. **Region Dense Video Captioning (RDCap):** In this task, a model must generate a detailed description of all events involving a specific subject of interest in a video.

## FGQA

The FGQA task is registered as `fgqa_test` in lmms-eval. Please follow the instruction below to evaluate on it.

1. **Download and Crop Videos:** To obtain the test videos, first download the original videos from their respective sources, then trim them into clips using the metadata in the [`test file`](https://huggingface.co/datasets/facebook/PLM-VideoBench/blob/main/fgqa/plm_fgqa_test.jsonl) for every sample. The `source_video_id`, `source_start_time` and `source_end_time` fields per sample can be used to obtain the video clips from each source dataset (which is specified in `source_dataset`). The original video sources refer to the following open-source datasets: [COIN](https://coin-dataset.github.io/), [Ego4d](https://ego4d-data.org/), [EgoExo4d](https://ego-exo4d-data.org/), [CrossTask](https://arxiv.org/abs/1903.08225) and [YouCook2](http://youcook2.eecs.umich.edu/), and [HT100M](https://www.di.ens.fr/willow/research/howto100m/). Make sure that after trimming the videos for each sample, you use the name indicated in the `video` field to save the trimmed video, and place all of them in a single, flat directory. 

2. **Modify [`_default_template_yaml`](_default_template_yaml)**: After downloading and trimming the FGQA videos, update the `video_base_dir` path in the default config file under `plm_fgqa` to point to the directory where you have placed the videos in a flat hierarchy.

Now you can use the `fgqa_test` task to run evaluation.

## SGQA

The SGQA task is registered as `sgqa_test` in lmms-eval. Please follow the instruction below to evaluate on it.

1. **Download Videos:** All the test videos can be downloaded from the from the huggingface repo [`sgqa-videos-download`](https://huggingface.co/datasets/facebook/PLM-VideoBench/tree/main/videos/sgqa).  

2. **Modify [`_default_template_yaml`](_default_template_yaml)**: After downloading the SGQA videos, update the `video_base_dir` path in the default config file under `plm_sgqa` to point to the directory where you have placed the videos in a flat hierarchy.

Now you can use the `sgqa_test` task to run evaluation.

## Video Grounding Tasks
The RCap, RTLoc and RDCap are registered as `rcap_test`, `rtloc_test` and `rdcap_test` in lmms-eval. Please follow the instruction below to evaluate on these tasks.

1. **Download Videos:** We have annotated SA-V (SAM-2) videos for these for these tasks which can be downloaded from the official website [`segment-anything-videos-download`](https://ai.meta.com/datasets/segment-anything-video-downloads). Please ONLY DOWNLOAD THE TAR FILE NAMED "videos_fps_6.tar", extrat and arrange as following.

```
├── video_base_dir
│   ├── sav_001*.mp4
│   ├── sav_001*.mp4
│   ├── ...

```

2. **Modify [`_default_template_yaml`](_default_template_yaml)**: After downloading and extracting the SA-V videos, update the `video_base_dir` path in the default config file under `plm_stc`.

Now you can use `rcap_test`, `rtloc_test` and `rdcap_test` tasks to run evaluation.

## Perception Lanuage Model (PLM)
We provide instruction to evaluate PLM on PLM-VideoBench.

### Prepare the Environment

```shell
# Install Perception Models
git clone https://github.com/facebookresearch/perception_models.git
cd perception_models

conda create --name perception-env python=3.12
conda activate perception-env

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124

# We use torchcodec for decoding videos into PyTorch tensors
conda install ffmpeg -c conda-forge
pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu124

pip install -e .


# Install lmms-eval
pip install lmms-eval
```

### Run Evaluation

```shell

# Use facebook/Perception-LM-1B for 1B parameters model and facebook/Perception-LM-3B for 3B parameters model.
CHECKPOINTS_PATH=facebook/Perception-LM-8B
OUTPUT_PATH=plmms_evaluation

# PLM-VideoBench Tasks
TASKS=fgqa_test,sgqa_test,rtloc_test,rcap_test,rdcap_test

accelerate launch --num_processes=<No. Of Available GPU> \
-m lmms_eval \
--model plm \
--model_args pretrained=$CHECKPOINTS_PATH \
--tasks $TASKS \
--batch_size 1 \
--log_samples \
--log_samples_suffix plm \
--output_path $OUTPUT_PATH
```
