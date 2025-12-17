# IllusionBench (Arshia Hemmat) 测试集构建文档

## 1. 数据集来源

- **Hugging Face 仓库**: `arshiahemmat/IllusionBench`
- **论文地址**: https://arshiahemmat.github.io/illusionbench/
- **任务类型**: 视觉错觉理解（Visual Illusion Understanding）

## 2. 数据集下载

### 2.1 下载脚本

```python
# download.py
from datasets import load_dataset
import os

# 设置镜像（可选）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载数据集
dataset = load_dataset("arshiahemmat/IllusionBench")

# 查看数据集结构
print(dataset)
```

### 2.2 执行下载

```bash
cd /data1/zwb
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hf
export HF_ENDPOINT="https://hf-mirror.com"
python download.py
```

### 2.3 数据集结构

下载后数据集包含 3 个 split：
- `Illusion_ICON`: 图标类视觉错觉图像
- `Illusion_LOGO`: Logo 类视觉错觉图像  
- `Illusion_IN`: ImageNet 类视觉错觉图像

每条数据的字段：
```python
{
    'image': <PIL.Image>,      # 图像数据（bytes）
    'image_name': str,         # 图像名称，包含 shape、scene、difficulty 信息
}
```

## 3. 数据分析

### 3.1 image_name 格式解析

`image_name` 字段格式：`{shape}-{index}-{scene}-{difficulty}-{version}.png`

示例：
- `animal-100-Sand_dune-Hard-1.1-64.png`
  - shape: `animal`
  - scene: `Sand_dune`
  - difficulty: `Hard`

### 3.2 难度分布分析

```python
from datasets import load_dataset
from collections import Counter

dataset = load_dataset("arshiahemmat/IllusionBench")

def parse_difficulty(image_name: str) -> str:
    name = image_name.lower()
    if "-easy-" in name:
        return "Easy"
    elif "-medium-" in name:
        return "Medium"
    elif "-hard-" in name:
        return "Hard"
    return "Unknown"

for split in ["Illusion_ICON", "Illusion_LOGO", "Illusion_IN"]:
    ds = dataset[split]
    diff_counts = Counter(parse_difficulty(row["image_name"]) for row in ds)
    print(f"\n{split}:")
    for diff, count in sorted(diff_counts.items()):
        print(f"  {diff}: {count}")
```

## 4. 测试集构建

### 4.1 采样策略

- 每个 split 采样 **1000** 个样本
- 保持原始难度比例（Easy/Medium/Hard）
- 使用固定随机种子确保可复现

### 4.2 采样脚本

```python
# make_arshia_testset.py
import os
import json
import random
from collections import defaultdict
from datasets import load_dataset, Dataset

BASE = "/data1/zwb/hf/datasets/arshia_illusionbench_test1000"
SPLITS = ["Illusion_ICON", "Illusion_LOGO", "Illusion_IN"]
DIFF_TOKENS = ["easy", "medium", "hard"]
SEED = 42
N_PER_SPLIT = 1000  # 每个 split 采样数量

def parse_difficulty(image_name: str) -> str:
    name = image_name.lower()
    for tok in DIFF_TOKENS:
        if f"-{tok}-" in name:
            return tok.capitalize()
    return "Unknown"

def proportional_allocation(counts: dict, total: int) -> dict:
    """按比例分配采样数量"""
    grand = sum(counts.values())
    if grand == 0:
        return {k: 0 for k in counts}
    
    alloc = {}
    remainder = []
    running = 0
    
    for k, c in counts.items():
        exact = c / grand * total
        floor_val = int(exact)
        alloc[k] = floor_val
        running += floor_val
        remainder.append((exact - floor_val, k))
    
    # 分配剩余配额
    remainder.sort(reverse=True)
    for _, k in remainder:
        if running >= total:
            break
        alloc[k] += 1
        running += 1
    
    return alloc

def build_index(ds, parse_fn):
    """构建难度索引"""
    idx_by_diff = defaultdict(list)
    for i, row in enumerate(ds):
        diff = parse_fn(row["image_name"])
        idx_by_diff[diff].append(i)
    return idx_by_diff

def sample_refs(idx_by_diff: dict, alloc: dict, rng: random.Random) -> list:
    """按比例采样"""
    refs = []
    for diff, indices in idx_by_diff.items():
        n = alloc.get(diff, 0)
        chosen = rng.sample(indices, min(n, len(indices)))
        refs.extend(chosen)
    return sorted(refs)

def materialize_subset(ds, indices: list) -> Dataset:
    """创建子数据集"""
    return ds.select(indices)

def main():
    os.makedirs(BASE, exist_ok=True)
    rng = random.Random(SEED)
    
    dataset = load_dataset("arshiahemmat/IllusionBench")
    summary = {}
    
    for split in SPLITS:
        ds = dataset[split]
        idx_by_diff = build_index(ds, parse_difficulty)
        
        # 统计原始分布
        counts = {d: len(v) for d, v in idx_by_diff.items()}
        
        # 计算采样分配
        alloc = proportional_allocation(counts, N_PER_SPLIT)
        
        # 采样
        indices = sample_refs(idx_by_diff, alloc, rng)
        ds_sub = materialize_subset(ds, indices)
        
        # 保存为 parquet
        out_name = f"{split.lower()}_test{N_PER_SPLIT}.parquet"
        out_path = os.path.join(BASE, out_name)
        ds_sub.to_parquet(out_path)
        
        summary[split] = {
            "original_counts": counts,
            "sampled_counts": alloc,
            "total_sampled": len(indices),
            "output_file": out_name
        }
        print(f"[{split}] Saved {len(indices)} samples to {out_path}")
    
    # 保存摘要
    with open(os.path.join(BASE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
```

### 4.3 执行采样

```bash
cd /data1/zwb
python make_arshia_testset.py
```

### 4.4 输出文件

```
/data1/zwb/hf/datasets/arshia_illusionbench_test1000/
├── illusion_icon_test1000.parquet   # ICON split 测试集
├── illusion_logo_test1000.parquet   # LOGO split 测试集
├── illusion_in_test1000.parquet     # IN split 测试集
└── summary.json                     # 采样统计摘要
```

## 5. lmms-eval 集成

### 5.1 任务配置文件

创建任务 YAML 文件，位于 `/data1/zwb/lmms-eval/lmms_eval/tasks/illusionbench/`

#### Shape 任务示例 (illusionbench_arshia_icon_shape_test.yaml)

```yaml
task: illusionbench_arshia_icon_shape_test
dataset_path: parquet
dataset_name: null
data_files:
  Illusion_ICON: /data1/zwb/hf/datasets/arshia_illusionbench_test1000/illusion_icon_test1000.parquet
test_split: Illusion_ICON
output_type: generate_until
process_docs: !function arshia_utils.illusionbench_arshia_process_docs
doc_to_visual: !function arshia_utils.illusionbench_arshia_doc_to_visual
doc_to_text: !function arshia_utils.illusionbench_arshia_doc_to_text_shape_icon
doc_to_target: !function arshia_utils.illusionbench_arshia_doc_to_target
generation_kwargs:
  max_new_tokens: 128
  do_sample: false
metric_list:
  - metric: shape_recall
    aggregation: !function arshia_utils.illusionbench_arshia_aggregate
    higher_is_better: true
process_results: !function arshia_utils.illusionbench_arshia_process_results_shape
```

#### Scene 任务示例 (illusionbench_arshia_icon_scene_test.yaml)

```yaml
task: illusionbench_arshia_icon_scene_test
dataset_path: parquet
dataset_name: null
data_files:
  Illusion_ICON: /data1/zwb/hf/datasets/arshia_illusionbench_test1000/illusion_icon_test1000.parquet
test_split: Illusion_ICON
output_type: generate_until
process_docs: !function arshia_utils.illusionbench_arshia_process_docs
doc_to_visual: !function arshia_utils.illusionbench_arshia_doc_to_visual
doc_to_text: !function arshia_utils.illusionbench_arshia_doc_to_text_scene
doc_to_target: !function arshia_utils.illusionbench_arshia_doc_to_target
generation_kwargs:
  max_new_tokens: 128
  do_sample: false
metric_list:
  - metric: scene_recall
    aggregation: !function arshia_utils.illusionbench_arshia_aggregate
    higher_is_better: true
process_results: !function arshia_utils.illusionbench_arshia_process_results_scene
```

### 5.2 工具函数 (arshia_utils.py)

关键函数：

```python
def parse_image_name(image_name: str) -> dict:
    """从 image_name 解析 shape 和 scene"""
    # 格式: {shape}-{index}-{scene}-{difficulty}-{version}.png
    # 例: animal-100-Sand_dune-Hard-1.1-64.png
    ...

def illusionbench_arshia_process_docs(dataset):
    """预处理数据集，解析 shape_gt 和 scene_gt"""
    ...

def illusionbench_arshia_doc_to_visual(doc):
    """解码图像字段"""
    ...

def illusionbench_arshia_doc_to_text_shape_icon(doc):
    """Shape 任务的 prompt（带候选选项）"""
    ...

def illusionbench_arshia_doc_to_text_scene(doc):
    """Scene 任务的 prompt（开放式问答）"""
    ...
```

### 5.3 运行评测

```bash
# Qwen2.5-VL-7B 评测
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=/data1/zwb/hf/models/Qwen2.5-VL-7B" \
  --tasks illusionbench_arshia_test \
  --batch_size 4 \
  --log_samples \
  --output_path ./logs/illusionbench_qwen25vl_test1000

# Bagel-7B 评测
python -m lmms_eval \
  --model bagel \
  --model_args "pretrained=/data1/zwb/hf/models/BAGEL-7B-MoT,show_thinking=false" \
  --tasks illusionbench_arshia_test \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/illusionbench_bagel_test1000
```

## 6. 评测指标

### 6.1 Shape Recall

- 将模型输出与 ground truth shape 进行文本匹配
- 归一化处理：小写、去除标点、去除多余空格
- 计算召回率：正确匹配数 / 总样本数

### 6.2 Scene Recall

- 将模型输出与 ground truth scene 进行文本匹配
- 同样进行归一化处理
- 计算召回率

## 7. 注意事项

1. **数据集格式**: 原始数据集的 `image` 字段是 bytes，需要解码为 PIL.Image
2. **Ground Truth**: shape 和 scene 的 ground truth 从 `image_name` 字段解析得到
3. **难度保持**: 采样时保持原始难度比例，确保测试集代表性
4. **Prompt 设计**: Shape 任务使用多选题格式，Scene 任务使用开放式问答

## 8. 参考链接

- IllusionBench 官网: https://arshiahemmat.github.io/illusionbench/
- Hugging Face 数据集: https://huggingface.co/datasets/arshiahemmat/IllusionBench
- lmms-eval 框架: https://github.com/EvolvingLMMs-Lab/lmms-eval

