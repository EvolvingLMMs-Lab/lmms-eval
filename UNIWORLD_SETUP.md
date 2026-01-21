# UniWorld Setup Guide

## ❌ 当前问题

遇到 `KeyError: 'text_config'` 错误，这是因为 UniWorld 的配置文件与标准 Qwen2.5-VL 格式不兼容。

## ✅ 解决方案

### 方案 1：下载模型到本地（推荐）

不要直接使用 HuggingFace 模型 ID，而是先下载到本地：

```bash
# 1. 安装 huggingface-cli
pip install huggingface-hub

# 2. 下载 UniWorld 模型到本地
huggingface-cli download LanguageBind/UniWorld-V1 --local-dir ./models/UniWorld-V1

# 3. 修改脚本使用本地路径
bash uniworld_general.sh "2" "chartqa100" "./logs/chartqa" "./models/UniWorld-V1"
```

### 方案 2：修复配置文件

如果模型已下载，手动修复配置文件：

```bash
# 编辑 config.json，确保包含以下字段
cd models/UniWorld-V1  # 或者 ~/.cache/huggingface/hub/...

# 在 config.json 中添加缺失的配置
# 需要确保有 text_config 和 vision_config 字段
```

### 方案 3：使用 trust_remote_code

已在代码中添加 `trust_remote_code=True` 和错误处理。

再次尝试运行：

```bash
bash uniworld_general.sh "2" "chartqa100" "./logs/chartqa"
```

## 📋 检查清单

如果还有问题，请检查：

### 1. UniWorld 仓库是否正确克隆？

```bash
ls -la UniWorld/UniWorld-V1/
# 应该看到：
#   - univa/
#   - README.md
#   - 其他文件
```

### 2. 依赖是否安装完整？

```bash
pip install transformers accelerate torch torchvision
pip install flash-attn  # 用于 flash_attention_2
pip install diffusers  # 用于 FLUX pipeline
```

### 3. transformers 版本

```bash
pip show transformers
# 应该是 >= 4.40.0
```

## 🔧 调试步骤

### 测试导入

```bash
python -c "from lmms_eval.models.simple.uniworld import UniWorld; print('✅ Import successful')"
```

### 查看详细错误

```bash
bash uniworld_general.sh "2" "chartqa100" "./logs/chartqa" 2>&1 | tee uniworld_error.log
```

### 检查 HuggingFace 缓存

```bash
ls ~/.cache/huggingface/hub/models--LanguageBind--UniWorld-V1/
```

## 🚨 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `KeyError: 'text_config'` | 配置文件不完整 | 下载完整模型到本地 |
| `restore_default_torch_dtype` | transformers 版本不兼容 | ✅ 已修复 |
| `flash_attention_2 not found` | flash-attn 未安装 | `pip install flash-attn` |
| `FLUX model not found` | diffusers 未安装 | `pip install diffusers` |

## 📞 需要更多帮助？

如果问题持续，请提供：

1. **完整错误日志**：
   ```bash
   bash uniworld_general.sh "2" "chartqa100" "./logs/chartqa" 2>&1 | tee error.log
   ```

2. **环境信息**：
   ```bash
   pip list | grep -E "(transformers|torch|accelerate)"
   ```

3. **UniWorld 目录结构**：
   ```bash
   tree -L 2 UniWorld/
   ```
