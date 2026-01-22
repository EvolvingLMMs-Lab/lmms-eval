#!/bin/bash
# 安装lmms-eval和UniWorld依赖，UniWorld版本优先

set -e  # 出错时退出

echo "=========================================="
echo "Installing Dependencies"
echo "=========================================="
echo ""

# 确认当前在正确的虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Error: No virtual environment activated!"
    echo "Please activate your virtual environment first:"
    echo "  source 11/bin/activate"
    exit 1
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# Step 1: 安装 lmms-eval 基础依赖
echo "=========================================="
echo "Step 1: Installing lmms-eval dependencies"
echo "=========================================="
cd /home/aiscuser/lmms-eval
uv pip install -e .
echo "✅ lmms-eval installed"
echo ""

# Step 2: 安装 UniWorld 依赖（会覆盖冲突的包）
echo "=========================================="
echo "Step 2: Installing UniWorld dependencies"
echo "=========================================="
cd /home/aiscuser/lmms-eval/UniWorld/UniWorld-V1

# 重要的包版本（UniWorld优先）
echo "Installing UniWorld-specific packages..."
uv pip install transformers==4.50.0 --force-reinstall
uv pip install accelerate==1.5.2 --force-reinstall  
uv pip install httpx==0.24.1 --force-reinstall
uv pip install diffusers==0.32.2 --force-reinstall
uv pip install torch==2.5.1 --force-reinstall
uv pip install torchvision==0.20.1 --force-reinstall

# 安装其他UniWorld依赖
uv pip install -r requirements.txt

echo "✅ UniWorld dependencies installed"
echo ""

# Step 3: 验证关键包版本
echo "=========================================="
echo "Step 3: Verifying package versions"
echo "=========================================="
echo "huggingface-hub: $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "accelerate: $(python -c 'import accelerate; print(accelerate.__version__)')"
echo "diffusers: $(python -c 'import diffusers; print(diffusers.__version__)')"
echo "httpx: $(python -c 'import httpx; print(httpx.__version__)')"
echo ""

echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test UniWorld model loading:"
echo "   cd /home/aiscuser/lmms-eval"
echo "   python test_uniworld_direct.py"
echo ""
echo "2. Run evaluation:"
echo "   bash g2u/uniworld.sh '0' 'chartqa100' './logs/chartqa'"
