# LSDBench: Long-video Sampling Dilemma Benchmark

**LSDBench** is a benchmark designed to evaluate the sampling efficiency of Vision-Language Models (VLMs) in long-video tasks. It addresses the "Sampling Dilemma," where low-density sampling risks missing critical information, and high-density sampling introduces redundancy, slowing inference.

**Key Resources**:  
- **Arxiv Paper**: [📖 Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma?](https://arxiv.org/abs/2503.12496)  
- **Huggingface Dataset**: [🤗 LSDBench](https://huggingface.co/datasets/TainU/LSDBench)


## Core Features

1. **LSDBench Dataset**:  
   - **QA Pairs**: 1304  
   - **Videos**: 400  
   - **Avg. Length**: 45.39 min/video  
   - **Focus**: Dense action sequences requiring high Necessary Sampling Density (NSD).  

2. **Proposed Frameworks**:
   - **Reasoning-Driven Hierarchical Sampling (RHS)**: Efficiently processes long videos by focusing on relevant segments.  
   - **Semantic-Guided Frame Selector (SGFS)**: Selects frames with high visual information content.

3. **Evaluation**:  
   Models are tested under three settings:  
   - **Only Text**: No visual input.  
   - **Oracle**: Annotated video segments as input.  
   - **Full Video**: Complete video input.  
