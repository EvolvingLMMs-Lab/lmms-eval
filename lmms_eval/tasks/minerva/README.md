## MINERVA
MINERVA consists of ~1.5K
challenging question-answer-decoy (QAD) sets for variable length videos. For
each question, we provide 5 answer choices, as well as detailed,
manually-annotated reasoning traces. Every question in MINERVA requires complex
reasoning using two or more skills
(for example numerical reasoning, temporal reasoning, spatial navigation).
Videos also span multiple domains (short films, sports, instructional videos
etc), with various video lengths (from 2 minutes to over 1.5 hours). The
hand-crafted, detailed reasoning trace accompanying each question outlines
the steps that are required to come to the correct answer.
These traces include timestamps where necessary to refer to relevant sections of
the video, and also describes key actions, objects, as well as outlines logical
reasoning steps. More details are provided in our
[arXiv](https://arxiv.org/abs/2505.00681) paper.

### Downloading the Data
We provide a json file that contains the YouTube IDs and annotations.

The json file contains the following fields:

- key: Unique identifier for each question
- video_id: YouTube URL
- question: Free-form question
- answer: Free-form answer
- answer_choice_{i}: Decoys for MCQ evaluation, i in range(0,4)
- answer_id: ID of the correct answer in the decoys
- reasoning: Detailed reasoning trace
- question type: A comma-separated list of multiple skills needed to answer the
question
- split: Coarse video domain
- category: Fine-grained video domain

[MINERVA json](https://storage.mtls.cloud.google.com/neptunedata/minerva.json)

### Citing this work
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
```latex
@article{minerva25,
  title={MINERVA: Evaluating Complex Video Reasoning},
  author={Nagrani, Arsha and Menon, Sachit and Iscen, Ahmet and Buch, Shyamal and Mehran, Ramin and Jha, Nilpa and Hauth, Anja and Zhu, Yukun and Vondrick, Carl and Sirotenko, Mikhail and Schmid, Cordelia and Weyand, Tobias},
  journal={arXiv preprint arXiv:2505.00681},
  year={2025}
}
```