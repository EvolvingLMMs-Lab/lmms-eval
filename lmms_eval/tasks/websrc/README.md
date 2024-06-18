# WebSRC

## Paper 

Title: WebSRC: A Dataset for Web-Based Structural Reading Comprehension

Abstract: https://arxiv.org/abs/2101.09465

Homepage: https://x-lance.github.io/WebSRC/#

WebSRC is a dataset for web-based structural reading comprehension.
Its full train/dev/test split contains over 400k questions across 6.4k webpages. 
This version of the dataset does not contain OCR or original HTML, it simply treats WebSRC as a image-and-text-based multimodal Q&A benchmark on webpage screenshots.

## Citation

```bibtex
@inproceedings{chen2021websrc,
  title={WebSRC: A Dataset for Web-Based Structural Reading Comprehension},
  author={Chen, Xingyu and Zhao, Zihan and Chen, Lu and Ji, Jiabao and Zhang, Danyang and Luo, Ao and Xiong, Yuxuan and Yu, Kai},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={4173--4185},
  year={2021}
}
```

## Groups & Tasks

### Groups 

- `websrc`: Evaluates `websrc-val` and generates a submission file for `websrc-test`.

### Tasks

- `websrc-val`: Given a question and a web page, predict the answer.
- `websrc-test`: Given a question and a web page, predict the answer. Ground truth is not provided for this task.

## Metrics

This task uses SQUAD-style evaluation metrics, of which F1 score over tokens is used. 
The orignal paper also uses Exact Match (EM) score, but this is not implemented here as that metric is more conducive for Encoder-only extraction models.

### F1 Score

F1 Score is the harmonic mean of precision and recall.
We calculate precision and recall at the token level, then compute the F1 score as normal using these values.

### Test Submission

When evaluaing on the test split, a prediction JSON will be compiled instead of metrics computed. 
Instructions for submission are available on the [WebSRC homepage](https://x-lance.github.io/WebSRC/#) and in their [Original GitHub Repo](https://github.com/X-LANCE/WebSRC-Baseline#obtain-test-result).