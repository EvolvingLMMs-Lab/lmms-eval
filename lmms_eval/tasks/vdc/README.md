# VDC

## Task Description

This repository contains an evaluation dataset designed for assessing the video detailed structure captioning performance of video models. The dataset includes ChatGPT-generated question-answer pairs based on the video descriptions in different aspects.

- GPT-4o Evaluation: The answers are evaluated using the prompts designed by Video-ChatGPT, which rates the responses based on the aforementioned dimensions with `gpt-4o-mini`.

## Groups & Tasks

### Tasks

- `vdc_camera`: Given a video, generate a detailed caption that thoroughly describes the camera work, including shot types, angles, camera movements, transitions between scenes, and any special effects used.
- `vdc_background`: Given a video, generate a detailed caption that describes the background, including objects, locations, weather conditions, time of day, and any dynamic elements in the scene.
- `vdc_detailed`: Given a video, generate a comprehensive caption that thoroughly describes all aspects of the scene, including narrative elements, character interactions, emotional tones, and intricate visual or auditory details.
- `vdc_main_object`: Given a video, generate a comprehensive caption that analyzes the primary subjects, detailing their actions, attributes, interactions, and movements across frames. Include variations in posture, facial expressions, and speed of movement throughout the scene.
- `vdc_short`: Given a video, generate a concise one-sentence caption that succinctly summarizes the main events and key elements of the scene without extensive detail.

## Model Performance Comparison

| **Model**            | **Camera** (Acc / Score) | **Short** (Acc / Score) | **Background** (Acc / Score) | **Main Object** (Acc / Score) | **Detailed** (Acc / Score) |
|----------------------|--------------------------|-------------------------|------------------------------|-------------------------------|-----------------------------|
| Vicuna-v1.5-7B       | 21.68 / 1.12             | 23.06 / 1.17            | 22.02 / 1.15                 | 22.64 / 1.16                  | 23.09 / 1.20                |
| Llama-3.1-8B         | 17.83 / 1.00             | 17.90 / 1.02            | 19.52 / 1.10                 | 19.57 / 1.10                  | 20.10 / 1.22                |
| Gemini-1.5 Pro       | 38.68 / 2.05             | 35.71 / 1.85            | 43.84 / 2.23                 | 47.32 / 2.41                  | 43.11 / 2.22                |
| LLaMA-VID            | 39.47 / 2.10             | 29.92 / 1.56            | 28.01 / 1.45                 | 31.24 / 1.59                  | 25.67 / 1.38                |
| Video-ChatGPT-7B     | 37.46 / 2.00             | 29.36 / 1.56            | 33.68 / 1.70                 | 30.47 / 1.60                  | 24.61 / 1.26                |
| MovieChat-7B         | 37.25 / 1.98             | 32.55 / 1.59            | 28.99 / 1.54                 | 31.97 / 1.64                  | 28.82 / 1.46                |
| VILA-7B              | 34.33 / 1.83             | 30.40 / 1.55            | 35.15 / 1.80                 | 33.38 / 1.72                  | 29.78 / 1.58                |
| Video-LLaVA-7B       | 37.48 / 1.97             | 30.67 / 1.63            | 32.50 / 1.70                 | 36.01 / 1.85                  | 27.36 / 1.43                |
| LLaVA-1.5-7B         | 38.38 / 2.04             | 28.61 / 1.51            | 34.86 / 1.79                 | 34.62 / 1.76                  | 33.43 / 1.73                |
| LongVA-7B            | 35.32 / 1.90             | 31.94 / 1.63            | 36.39 / 1.85                 | 40.95 / 2.11                  | 27.91 / 1.48                |
| LLaVA-1.5-13B        | 38.97 / 2.07             | 30.89 / 1.60            | 34.79 / 1.78                 | 36.27 / 1.84                  | 33.00 / 1.74                |
| LLaVA-NeXT-V7B       | 39.73 / 2.10             | 30.63 / 1.60            | 36.54 / 1.88                 | 36.54 / 1.88                  | 33.84 / 1.77                |
| LLaVA-1.6-7B         | 36.50 / 1.93             | 31.91 / 1.65            | 37.58 / 1.92                 | 36.03 / 1.85                  | 36.47 / 1.89                |
| LLaVA-1.6-13B        | 35.61 / 1.86             | 31.90 / 1.66            | 38.90 / 1.99                 | 36.65 / 1.87                  | 36.18 / 1.89                |
| ShareGPT4Video-8B    | 33.28 / 1.76             | 39.08 / 1.94            | 35.77 / 1.81                 | 37.12 / 1.89                  | 35.62 / 1.84                |
| LLaVA-OV-7B          | 37.82 / 2.02             | 32.58 / 1.70            | 37.43 / 1.92                 | 38.21 / 1.96                  | 41.20 / 2.13                |
| InternVL-2-8B        | 39.08 / 2.11             | 33.02 / 1.74            | 37.47 / 1.89                 | 44.16 / 2.22                  | 34.89 / 1.82                |
| **AURORACAP-7B**     | **43.50 / 2.27**         | **32.07 / 1.68**        | **35.92 / 1.84**             | **39.02 / 1.97**              | **41.30 / 2.15**            |
  
## Citation

```bibtex
@article{chai2024auroracap,
  title={AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark},
  author={Chai, Wenhao and Song, Enxin and Du, Yilun and Meng, Chenlin and Madhavan, Vashisht and Bar-Tal, Omer and Hwang, Jeng-Neng and Xie, Saining and Manning, Christopher D},
  journal={arXiv preprint arXiv:2410.03051},
  year={2024}
}
```