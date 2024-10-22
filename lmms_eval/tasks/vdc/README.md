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
  
## Citation

```bibtex
@article{chai2024auroracap,
  title={AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark},
  author={Chai, Wenhao and Song, Enxin and Du, Yilun and Meng, Chenlin and Madhavan, Vashisht and Bar-Tal, Omer and Hwang, Jeng-Neng and Xie, Saining and Manning, Christopher D},
  journal={arXiv preprint arXiv:2410.03051},
  year={2024}
}
```