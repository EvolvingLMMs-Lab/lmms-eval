# Tools

Utility scripts for lmms-eval development and dataset preparation.

## Scripts

### `get_split_zip.py`

Split large ZIP files into smaller parts for hosting services with file size limits.

```bash
# Split into 5GB parts (default)
python tools/get_split_zip.py dataset.zip ./output/

# Split into 2GB parts
python tools/get_split_zip.py dataset.zip ./output/ --max-size 2GB
```

### `regression.py`

Run regression tests across git branches to compare model performance.

```bash
python tools/regression.py --tasks ocrbench,mmmu_val --limit 8
```

## Notebooks

### `make_image_hf_dataset.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EvolvingLMMs-Lab/lmms-eval/blob/main/tools/make_image_hf_dataset.ipynb)

Tutorial for creating properly formatted Hugging Face datasets with images. Demonstrates the complete workflow from raw data to HF Hub upload.

---

## Archived Modules

The following modules were removed during cleanup. They can be found in the `main` branch if needed:

- **`lite/`** - LMMs-Eval Lite for dataset core-set selection using embedding-based sampling
- **`live_bench/`** - Separate package for LiveBench data generation and website screenshot capture
