
from huggingface_hub import snapshot_download
import os

# Download to /scratch directory (largest available space)
save_dir = "/scratch/models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

print(f"Downloading BAGEL-7B-MoT to: {save_dir}")
print(f"This will take ~15-20GB of space")
print("=" * 60)

# Create directory if not exists
os.makedirs(save_dir, exist_ok=True)

snapshot_download(
    cache_dir=cache_dir,
    local_dir=save_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

print("=" * 60)
print(f"Download complete! Model saved to: {save_dir}")
print("\nTo use this model, run:")
print(f"  python test_bagel_easi.py --model_path {save_dir} --limit 10")
