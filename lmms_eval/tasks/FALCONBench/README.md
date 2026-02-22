# FALCONBench

FALCONBench is a comprehensive benchmark designed to evaluate the capabilities of multimodal large language models (MLLMs) in understanding and reasoning over **one-hour long videos**. It is part of the [FALCONEye project](https://cplou99.github.io/FALCONEye/), which aims to push the boundaries of video-language understanding by introducing challenging tasks that require both visual and temporal comprehension.

## Context

Traditional video-language benchmarks often focus on short clips or simple question-answering tasks. **FALCONBench stands out by using full-length, one-hour videos** from diverse domains (sports, movies, walking tours). This presents unique challenges:

- **Sparse frame sampling is insufficient:** Many answers are contained within a tiny temporal window and a small spatial region of a few frames. Sampling only a few frames from a long video will likely miss critical information.
- **Heavy token compression fails:** Compressing all frames into a small set of tokens loses the fine-grained details needed, as answers may depend on subtle cues in a specific moment.

Thus, FALCONBench requires models to reason over extended temporal contexts and to localize answers both in time and space.

For more details and the motivation behind FALCONBench, see the [FALCONEye project page](https://cplou99.github.io/FALCONEye/).

## Key Features

- **One-Hour Long Video Reasoning:** Unlike most benchmarks, FALCONBench uses one-hour-long videos, challenging models to reason over extended temporal contexts.
- **Temporal Localization:** Tasks require models to specify the exact time window in which an answer is observed.
- **Diverse Domains:** Videos are sourced from SoccerNet, MovieChat-1K, and Walking Tours, covering sports, movies, and real-world scenarios.
- **Multiple Task Types:** Includes both multiple-choice (MCQ) and open-ended (OQ) questions, with and without temporal localization.
- **Automatic Evaluation:** Open-ended answers are evaluated using GPT-4o-mini for semantic correctness.

## Setup Instructions


Before using FALCONBench, you must complete the following steps:

**Note:** The benchmark requires the `soccernet` Python package. You can install it via pip:

```bash
pip install soccernet
```

1. **Download Video Data**
	 - **SoccerNet:**  
		 - Fill out the [SoccerNet NDA form](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform).
		 - Save the password sent to your email as the environment variable `SOCCERNET_PWD`.
	 - **MovieChat-1K:**  
		 - Request access at [MovieChat-1K on HuggingFace](https://huggingface.co/datasets/Enxin/MovieChat-1K_train).
	 - **Walking Tours:**  
		 - These videos are already included in the Huggingface repository.

2. **Set Environment Variables**
	 - `SOCCERNET_PWD`: Password for SoccerNet video download.
	 - `OPENAI_API_KEY`: Required for open-ended question evaluation (OQ tasks).

	 Example (Linux):
	 ```bash
	 export SOCCERNET_PWD=your_soccernet_password
	 export OPENAI_API_KEY=your_openai_api_key
	 ```

3. **Download and Organize Videos**
	 - The first time you run the benchmark, the script will download the videos from the different sources and organize them in dataset_kwargs['cache_dir']/full_videos directory if they are not already present.

## Task Overview

FALCONBench includes four main tasks:

- **FALCONBench_mcq:**  
	Multiple-choice questions about video content. The model selects the correct option (A, B, C, ...).

- **FALCONBench_mcq_temploc:**  
	Multiple-choice with temporal localization. The model selects the correct option and specifies the time window (in seconds) where the answer is observed.

- **FALCONBench_oq:**  
	Open-ended questions. The model generates a free-form answer, which is evaluated for semantic correctness.

- **FALCONBench_oq_temploc:**  
	Open-ended with temporal localization. The model generates an answer and specifies the relevant time window.

### Example Output Format for Temporal Localization Tasks

The model should return:

```json
{
	"response": "A person running",
	"temporal_window": [105, 140]
}
```

## Example: Running FALCONBench with LLaVA-Video

To launch the benchmark using the LLaVA-Video model, use the following command:

```bash
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
		--model llava_vid \
		--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
		--tasks FALCONBench_mcq \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix falconbench-llava_vid_7B \
		--output_path ./logs/
```

**Note:** In the FALCONEye paper, results for small 7B VLMs are reported only for the MCQ and OQ tasks (without temporal localization) because these models struggle to output a json dictionary with both the answer and the temporal window, leading to a significant drop in accuracy when required to do so.

Replace `--tasks FALCONBench_mcq` with any of the other tasks (`FALCONBench_mcq_temploc`, `FALCONBench_oq`, `FALCONBench_oq_temploc`) as needed.

## Citation

If you use FALCONBench in your research, please cite the following:

```bibtex
@article{plou2025falconeye,
			title={FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs}, 
			author={Carlos Plou and Cesar Borja and Ruben Martinez-Cantin and Ana C. Murillo},
			booktitle={Proceedings of Winter Conference on Applications of Computer Vision},
			year={2026},
			eprint={2503.19850},
			archivePrefix={arXiv},
			primaryClass={cs.CV},
			url={https://arxiv.org/abs/2503.19850},
}
```

---
