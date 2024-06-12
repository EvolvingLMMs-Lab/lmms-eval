# SceenSpot

## GUI Grounding Benchmark: ScreenSpot

ScreenSpot is an evaluation benchmark for GUI grounding, comprising over 1200 instructions from iOS, Android, macOS, Windows and Web environments, along with annotated element types (Text or Icon/Widget).


## Groups

- `screenspot`: This group bundles both the original grounding task and the new instruction generation task.

## Tasks
- `screenspot_rec_test`: the original evaluation of `{img} {instruction} --> {bounding box}` called grounding or Referring Expression Completion (REC);
- `screenspot_reg_test`: the new evaluation of `{img} {bounding box} --> {instruction}` called instruction generation or Referring Expression Generation (REG).

### REC Metrics

REC/Grounding requires that a model outputs a bounding box for the target element in the image. The evaluation metrics are:
- `IoU`: Intersection over Union (IoU) between the predicted bounding box and the ground truth bounding box. 
- `ACC@IoIU`: We use `IoU` to create `ACC@IoU` metrics at different IoU thresholds where an output with an IoU above the threshold is considered correct.
- `CENTER ACC`: The predicted bounding box is considered correct if the center of the predicted bounding box is within the ground truth bounding box. This is what's reported in the paper.

### REG Metrics

REG/Generation requires that a model outputs the instruction that describes the target element in the image. Currently, this element will be highlighted in red in the image. The evaluation metrics are:
- `CIDEr`: The CIDEr metric is used to evaluate the quality of the generated instruction. As the paper doesn't consider this task, we have selected this metric as a standard for evaluating the quality of the generated instruction. This matches with what other works like ScreenAI have done for instruction generation for RICO datasets.

## Baseline Scores

As a Baseline, here is how LLaVA-v1.5-7b performs on the ScreenSpot dataset:
- `IoU`: 0.051
- `ACC@0.1`: 0.195
- `ACC@0.3`: 0.042
- `ACC@0.5`: 0.006
- `ACC@0.7`: 0.000
- `ACC@0.9`: 0.000
- `CENTER ACC`: 0.097
- `CIDEr`: 0.097

## References 

- ArXiv: [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/abs/2401.10935)
- GitHub: [njucckevin/SeeClick](https://github.com/njucckevin/SeeClick)

```bibtex
@misc{cheng2024seeclick,
      title={SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents}, 
      author={Kanzhi Cheng and Qiushi Sun and Yougang Chu and Fangzhi Xu and Yantao Li and Jianbing Zhang and Zhiyong Wu},
      year={2024},
      eprint={2401.10935},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```
