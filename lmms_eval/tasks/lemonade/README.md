# LEMONADE

## Task Description  

**LEMONADE** (Language models Evaluation of MOtion aNd Action-Driven Enquiries) is a QA benchmark extracted from the **EPFL-Smart-Kitchen-30** dataset (see [arXiv](https://arxiv.org/abs/2506.01608)). It consists of **36,521 closed-ended QA pairs** linked to egocentric video clips.  

Questions are organized into three groups and six subcategories:  

- **Behavior Understanding**  
  - *Perception*: recognizing perceived actions  
  - *Reasoning*: reasoning over unseen behaviors  
- **Long-term Understanding**  
  - *Summarization*: summarizing over longer clips  
  - *Session Properties*: inferring session-level information  
- **Motion & Biomechanics**  
  - *Physical Attributes*: inferring hand shapes, joint angles, etc.  
  - *Kinematics*: inferring trajectory velocities  

The benchmark was evaluated using **`lmms-eval`** in the associated publication.  


## Implementation  

- **utils.py**: Handles data loading from Hugging Face, video loading, answer parsing, and metric evaluation.  
- **lemonade.yaml**: Contains the default prompts and evaluation settings.

When running LEMONADE through `lmms-eval`, the data is automatically downloaded. For direct dataset access, please refer to [Hugging Face](https://huggingface.co/datasets/amathislab/LEMONADE) or [Zenodo](https://zenodo.org/records/15535461).  

Performance is evaluated in terms of accuracy against the ground truth, with results reported overall as well as per category and subcategory.

## Citation  

If you use **LEMONADE**, please cite:  

```bibtex
@misc{bonnetto2025epflsmartkitchen,
      title={EPFL-Smart-Kitchen-30: Densely annotated cooking dataset with 3D kinematics to challenge video and language models}, 
      author={Andy Bonnetto and Haozhe Qi and Franklin Leong and Matea Tashkovska and Mahdi Rad and Solaiman Shokur and Friedhelm Hummel and Silvestro Micera and Marc Pollefeys and Alexander Mathis},
      year={2025},
      eprint={2506.01608},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.01608}, 
}
```