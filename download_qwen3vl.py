#!/usr/bin/env python3
"""
下载Qwen3-VL-30B-A3B-Instruct模型到默认缓存路径
默认路径: ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/
"""
import os
from pathlib import Path

# 使用镜像源加速下载（可选，如果HuggingFace访问慢）
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("开始下载 Qwen3-VL-30B-A3B-Instruct 模型")
print("=" * 60)

# 确认缓存路径
cache_dir = Path.home() / ".cache" / "huggingface"
print(f"缓存目录: {cache_dir}")
print(f"模型将保存到: {cache_dir}/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/")
print("=" * 60)

try:
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
    
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    
    print("\n📥 正在下载模型（这可能需要一些时间）...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu",  # 下载时只用CPU，避免占用GPU
    )
    
    print("\n📥 正在下载处理器...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("\n✅ 模型下载完成！")
    
    # 验证文件完整性
    model_path = cache_dir / "hub" / "models--Qwen--Qwen3-VL-30B-A3B-Instruct"
    if model_path.exists():
        import subprocess
        result = subprocess.run(
            f"ls {model_path}/snapshots/*/model-*.safetensors | wc -l",
            shell=True,
            capture_output=True,
            text=True
        )
        num_files = int(result.stdout.strip())
        print(f"✅ 发现 {num_files} 个模型分片文件")
        if num_files == 13:
            print("✅ 模型文件完整！")
        else:
            print(f"⚠️  警告：预期13个文件，但只找到{num_files}个")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n如果是网络问题，请尝试：")
    print("1. 使用VPN或代理")
    print("2. 或者手动从 https://hf-mirror.com 下载")
    exit(1)

print("\n" + "=" * 60)
print("现在可以运行 lmms-eval 了！")
print("=" * 60)

