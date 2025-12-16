# Azure TRAPI 使用指南

使用Azure AD认证的Azure OpenAI模型集成到lmms-eval框架。

## 🔑 前置条件

### 1. 安装依赖
```bash
uv add azure-identity
```

### 2. Azure认证设置

需要先通过Azure CLI登录：
```bash
az login
```

或者使用Managed Identity（在Azure VM/容器中自动可用）

## 🚀 使用方法

### 基本使用

```bash
python -m lmms_eval \
    --model azure_trapi \
    --tasks mme \
    --batch_size 1 \
    --output_path ./logs/
```

### 自定义配置

```bash
# 通过model_args配置
python -m lmms_eval \
    --model azure_trapi \
    --model_args deployment=gpt-4o_2024-11-20,temperature=0.7,max_new_tokens=2048 \
    --tasks mathvista \
    --output_path ./logs/
```

### 通过环境变量配置

```bash
# 设置环境变量
export TRAPI_INSTANCE="gcr/shared"
export TRAPI_DEPLOYMENT="gpt-4o_2024-11-20"
export TRAPI_API_VERSION="2024-10-21"
export TRAPI_SCOPE="api://trapi/.default"

# 运行评测
python -m lmms_eval \
    --model azure_trapi \
    --tasks mmbench \
    --output_path ./logs/
```

### 启用缓存（节省API调用）

```bash
python -m lmms_eval \
    --model azure_trapi \
    --model_args continual_mode=True,response_persistent_folder=./cache/azure \
    --tasks mme \
    --output_path ./logs/
```

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `deployment` | `gpt-4o_2024-11-20` | Azure OpenAI deployment名称 |
| `instance` | `gcr/shared` | TRAPI实例路径 |
| `api_version` | `2024-10-21` | Azure OpenAI API版本 |
| `scope` | `api://trapi/.default` | Azure AD认证scope |
| `timeout` | `120` | API超时时间（秒） |
| `max_retries` | `5` | 失败重试次数 |
| `continual_mode` | `False` | 是否启用响应缓存 |
| `max_new_tokens` | `1024` | 最大生成token数 |
| `temperature` | `0.0` | 生成温度 |

## 🔒 认证方式

使用 **ChainedTokenCredential** 进行认证，按顺序尝试：

1. **Azure CLI Credential** - 本地开发使用
   ```bash
   az login
   ```

2. **Managed Identity** - Azure环境中自动可用
   - Azure VM
   - Azure Container Instances
   - Azure App Service
   - Azure Functions

## 📊 与其他API模型的对比

| 特性 | azure_trapi | gpt4v | openai_compatible |
|------|-------------|-------|-------------------|
| **认证方式** | Azure AD | API Key | API Key |
| **适用场景** | Microsoft内部 | 公开OpenAI | 自托管/兼容API |
| **免密码** | ✅ | ❌ | ❌ |
| **企业安全** | ✅ 高 | 中 | 中 |

## 🎯 实际使用示例

### 评测MathVista
```bash
python -m lmms_eval \
    --model azure_trapi \
    --model_args deployment=gpt-4o_2024-11-20,continual_mode=True \
    --tasks mathvista_testmini \
    --limit 100 \
    --output_path ./logs/azure_mathvista/
```

### 评测MME
```bash
python -m lmms_eval \
    --model azure_trapi \
    --model_args temperature=0.0,max_new_tokens=512 \
    --tasks mme \
    --output_path ./logs/azure_mme/
```

### 批量评测多个任务
```bash
python -m lmms_eval \
    --model azure_trapi \
    --model_args continual_mode=True,response_persistent_folder=./cache \
    --tasks mme,mmbench,mathvista_testmini \
    --output_path ./logs/azure_batch/
```

## 🐛 故障排查

### 1. 认证失败
```
Error: Failed to setup Azure AD authentication
```
**解决方案**：
```bash
# 重新登录Azure CLI
az login

# 验证登录状态
az account show
```

### 2. Deployment不存在
```
Error: The API deployment for this resource does not exist
```
**解决方案**：检查deployment名称是否正确
```bash
export TRAPI_DEPLOYMENT="gpt-4o_2024-11-20"  # 确保名称正确
```

### 3. 权限不足
```
Error: Insufficient permissions
```
**解决方案**：确保Azure账号有访问TRAPI的权限

### 4. 超时错误
```
Error: Request timeout
```
**解决方案**：增加timeout时间
```bash
python -m lmms_eval \
    --model azure_trapi \
    --model_args timeout=300 \
    --tasks ...
```

## 💡 最佳实践

1. **使用缓存**：启用 `continual_mode=True` 避免重复API调用
2. **设置合理超时**：根据任务复杂度调整 `timeout`
3. **控制并发**：API模型不支持batch，使用 `--batch_size 1`
4. **监控成本**：使用 `--limit` 参数限制样本数量进行测试

## 🔗 相关链接

- [Azure OpenAI文档](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure Identity库](https://learn.microsoft.com/en-us/python/api/azure-identity)
- [TRAPI服务](https://trapi.research.microsoft.com)
