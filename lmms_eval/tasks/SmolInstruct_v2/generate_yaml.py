import os
import glob

# ================= 路径配置 =================
# 1. JSONL 数据所在目录
jsonl_dir = "/mnt/innovator/zyj/smol/processed_smol_test_data"

# 2. YAML 输出目录
output_yaml_dir = "/mnt/innovator/code/yangboxue/lmms-eval/lmms_eval/tasks/SmolInstruct_v2"

# ================= 任务配置 =================
TASKS = (
    'forward_synthesis', 'retrosynthesis', 'molecule_captioning', 'molecule_generation',
    'name_conversion-i2f', 'name_conversion-i2s', 'name_conversion-s2f', 'name_conversion-s2i',
    'property_prediction-esol', 'property_prediction-lipo', 'property_prediction-bbbp',
    'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider',
)

# 仅控制最大长度，其他参数保持默认 (Temperature=0.0)
TASKS_GENERATION_SETTINGS = {
    'retrosynthesis': {'max_new_tokens': 960},
    'name_conversion-i2f': {'max_new_tokens': 20},
    'name_conversion-s2f': {'max_new_tokens': 20},
    'property_prediction-esol': {'max_new_tokens': 20},
    'property_prediction-lipo': {'max_new_tokens': 20},
    'property_prediction-bbbp': {'max_new_tokens': 20},
    'property_prediction-clintox': {'max_new_tokens': 20},
    'property_prediction-hiv': {'max_new_tokens': 20},
    'property_prediction-sider': {'max_new_tokens': 20},
}
DEFAULT_MAX_NEW_TOKENS = 1024 

# ================= 生成逻辑 =================
os.makedirs(output_yaml_dir, exist_ok=True)
print(f"开始根据 utils.py 逻辑生成 YAML 文件...")

generated_tasks = []

for task_name_raw in TASKS:
    # 1. 确定 JSONL 文件路径
    jsonl_path = os.path.join(jsonl_dir, f"{task_name_raw}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"⚠️ 警告: 找不到 {jsonl_path}，跳过。")
        continue

    # 2. 获取参数
    settings = TASKS_GENERATION_SETTINGS.get(task_name_raw, {})
    max_new_tokens = settings.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS)

    # 3. 构造 YAML 内容
    # 这里的关键是 metric_list 全部指向 smol_metrics 和 smol_aggregate_results
    # 这样 lmms-eval 会把所有样本传给你的 aggregate 函数，由你在 Python 里分发
    
    task_name_yaml = f"smol_{task_name_raw}"
    
    yaml_content = f"""task: {task_name_yaml}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {jsonl_path}

output_type: generate_until
test_split: test

# 对应你在 utils.py 中定义的函数
doc_to_visual: !function utils.smol_doc_to_visual
doc_to_text: !function utils.smol_doc_to_text
doc_to_target: !function utils.smol_doc_to_target

generation_kwargs:
  max_new_tokens: {max_new_tokens}
  temperature: 0.0
  do_sample: false

# 对应你在 utils.py 中的后处理和聚合函数
process_results: !function utils.smol_process_results

metric_list:
  - metric: smol_metrics
    aggregation: !function utils.smol_aggregate_results
    higher_is_better: true
"""

    # 4. 写入子任务 YAML
    output_path = os.path.join(output_yaml_dir, f"{task_name_yaml}.yaml")
    with open(output_path, "w") as f:
        f.write(yaml_content)
    
    generated_tasks.append(task_name_yaml)
    print(f"  -> 生成: {task_name_yaml}.yaml")

# 5. 生成 Group YAML (SmolInstruct_v2)
GROUP_NAME = "SmolInstruct_v2"
group_content = f"""group: {GROUP_NAME}
task:
{chr(10).join(['  - ' + t for t in generated_tasks])}
"""

group_output_path = os.path.join(output_yaml_dir, f"_{GROUP_NAME}.yaml")
with open(group_output_path, "w") as f:
    f.write(group_content)

print(f"\n✅ 全部完成！")
print(f"Group 配置文件: {group_output_path}")
print("请确保你的 utils.py 和 __init__.py 位于同一目录下。")

# 5. 生成 Group YAML (SmolInstruct_v2)
GROUP_NAME = "SmolInstruct_v2"
group_content = f"""group: {GROUP_NAME}
task:
{chr(10).join(['  - ' + t for t in generated_tasks])}
"""

group_output_path = os.path.join(output_yaml_dir, f"_{GROUP_NAME}.yaml")
with open(group_output_path, "w") as f:
    f.write(group_content)