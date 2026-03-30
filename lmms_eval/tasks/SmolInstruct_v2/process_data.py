import json
import os
import pandas as pd

# ================= 配置区域 =================
# 1. 基础 snapshot 路径
snapshot_path = "/mnt/innovator/zyj/.cache/huggingface/hub/datasets--osunlp--SMolInstruct/snapshots/b3e028007c6abc043f62ed0aef4824dc9e86bacc"

root_dir = os.path.join(snapshot_path, "extracted_data")

# 输出目录
output_dir = "./processed_smol_test_data"
os.makedirs(output_dir, exist_ok=True)

TASKS = (
    'forward_synthesis', 'retrosynthesis', 'molecule_captioning', 'molecule_generation',
    'name_conversion-i2f', 'name_conversion-i2s', 'name_conversion-s2f', 'name_conversion-s2i',
    'property_prediction-esol', 'property_prediction-lipo', 'property_prediction-bbbp',
    'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider',
)

TASK_CATEGORY_MAP = {
    'molecule_captioning': 'captioning',
    'molecule_generation': 'generation',
    'forward_synthesis': 'generation',
    'retrosynthesis': 'generation',
    'name_conversion-i2f': 'translation',
    'name_conversion-i2s': 'translation',
    'name_conversion-s2f': 'translation',
    'name_conversion-s2i': 'translation',
    'property_prediction-esol': 'regression',
    'property_prediction-lipo': 'regression',
    'property_prediction-bbbp': 'classification',
    'property_prediction-clintox': 'classification',
    'property_prediction-hiv': 'classification',
    'property_prediction-sider': 'classification',
}
# ===========================================

print(f"数据根目录已设置为: {root_dir}")
if not os.path.exists(root_dir):
    print("❌ 错误：路径不存在！请检查文件夹名是否真的是 extracted_data")
    exit(1)

print(f"开始处理 Test 集数据，结果将保存至: {output_dir}")

for task in TASKS:
    task_type = TASK_CATEGORY_MAP.get(task, 'unknown')
    
    # 1. 读取 Sample 索引
    sample_file = os.path.join(root_dir, 'sample', 'instruction_tuning', 'test', task + '.json')
    
    if not os.path.exists(sample_file):
        print(f"⚠️ 跳过 {task}: 找不到索引文件 ({sample_file})")
        continue

    print(f"正在处理: {task} | 类型: {task_type}")

    with open(sample_file, 'r') as f:
        sample_record = json.load(f)
        template_name = sample_record['template_name']
        samples = sample_record['samples']

    # 2. 读取 Template
    template_file = os.path.join(root_dir, 'template', template_name, task + '.json')
    if not os.path.exists(template_file):
        print(f"  ❌ 错误：找不到模板文件 {template_file}")
        continue
        
    with open(template_file, 'r') as f:
        templates = json.load(f)

    # 3. 读取 Raw Data
    raw_file = os.path.join(root_dir, 'raw', 'test', task + '.jsonl')
    if not os.path.exists(raw_file):
        print(f"  ❌ 错误：找不到原始数据文件 {raw_file}")
        continue
        
    raw_data = []
    with open(raw_file, 'r') as f:
        for line in f:
            raw_data.append(json.loads(line))
            
    # 4. 读取 Core Tags
    core_tag_file = os.path.join(root_dir, 'core_tag', task + '.json')
    input_tag_L = None
    if os.path.exists(core_tag_file):
        with open(core_tag_file, 'r') as f:
            core_tags = json.load(f)
        input_tag_L, input_tag_R = core_tags['input']
    else:
        print(f"  ℹ️  未找到 core_tag 文件，将不添加 tag。")

    processed_rows = []
    for item in samples:
        idx = item['idx']
        template_id = item['template_id']
        
        # 防止索引越界（虽然通常不会）
        if idx >= len(raw_data):
            continue

        data_item = raw_data[idx]
        raw_input = data_item['input']
        
        # 拼接 Tag
        if input_tag_L:
            full_input_str = f"{input_tag_L} {raw_input} {input_tag_R}"
        else:
            full_input_str = raw_input
            
        template = templates[template_id]
        final_input = template['input'].replace('<INPUT>', full_input_str)
        
        raw_output = data_item['output']
        if isinstance(raw_output, dict):
            raw_output = raw_output[item['target']]
        
        final_output = str(raw_output)

        processed_rows.append({
            "question": final_input,
            "answer": final_output,
            "task": task,
            "task_type": task_type,
            "id": f"{task}_{idx}"
        })

    save_path = os.path.join(output_dir, f"{task}.jsonl")
    df = pd.DataFrame(processed_rows)
    df.to_json(save_path, orient="records", lines=True, force_ascii=False)
    print(f"  -> 已保存 {len(df)} 条数据")

print("\n全部完成！")