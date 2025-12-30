# PR Code Review - 快速参考

## 一行命令

```bash
# 审查单个 PR
Review PR #123

# 审查所有打开的 PR
Review all open PRs

# 审查并合并符合条件的 PR
Review all PRs and merge qualified ones
```

## 评分系统（0-5 分）

| 分数 | 报告？ | 含义 |
|------|--------|------|
| 0 | ❌ | 误报 |
| 1 | ❌ | 低可信度 |
| 2 | ❌ | 不确定 |
| 3 | ❌ | 中等（不紧急）|
| 4 | ✅ | **高可信度** |
| 5 | ✅ | **关键问题** |

只报告 >= 4 分的问题。

## 5 个审查维度

1. **CLAUDE.md 合规性** - 代码风格、测试、包管理
2. **Bug 扫描** - 逻辑错误、运行时错误
3. **Git 历史** - 违反模式、重复 bug
4. **PR 模式** - 之前 review 的适用建议
5. **代码注释** - TODO、警告、文档模式

## 会跳过的 PR

- ❌ 已关闭或已合并
- ❌ Draft PR
- ❌ 已有 Claude Code review
- ❌ 描述不完整
- ❌ 自动化 PR（trivial）

## 不会报告的问题

- 预存问题（PR 未修改的行）
- Linter/TypeChecker 可捕获的
- CLAUDE.md 未要求的风格问题
- 小问题、吹毛求疵
- 已用 lint ignore 注释忽略的

## 输出格式

### 有问题
```markdown
### Code review

Found 3 issues:

1. {描述} ({CLAUDE.md 引用 或 bug 原因})
   {带完整 SHA 的代码链接}

2. ...
```

### 无问题
```markdown
### Code review

No issues found. Checked for bugs and CLAUDE.md compliance.
```

## 时间估算

- 单个 PR：2-3 分钟
- 10 个 PR：20-30 分钟
- 20 个 PR：40-60 分钟

## 代码链接格式

```
https://github.com/{owner}/{repo}/blob/{完整SHA}/{文件}#L{起始}-L{结束}

示例：
https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/831d6d277e0903c434d3e4bd1e2cf9f58757f207/lmms_eval/tasks/ovobench/utils.py#L102-L103
```

## 常见场景

### 场景 1：日常审查
```
每天早上：Review all open PRs
```

### 场景 2：合并前审查
```
合并前：Review PR #456
等待无问题或修复后再合并
```

### 场景 3：批量审查
```
周五下午：Review all open PRs
周末让团队修复问题
```

## 调试

### Skill 没触发
确保消息包含：
- "review"
- "PR" 或 "pull request"
- PR 编号

### 太多问题
正常！评分会过滤到高质量问题。
如仍太多，提高阈值到 5。

### 太少问题
正常！说明代码质量好。
如需更严格，降低阈值到 3。

## 团队协作

### 提交 Skill
```bash
git add .claude/skills/pr-code-review/
git commit -m "feat: add PR review skill"
git push
```

### 自定义
编辑 `SKILL.md`:
- 修改评分阈值（Step 7）
- 添加自定义检查（Step 5）
- 调整误报规则

## 性能提示

✅ **DO**:
- 并行审查多个 PR
- 信任 agent 评分
- 定期审查（每天/每周）

❌ **DON'T**:
- 手动检查每个问题（信任评分）
- 降低阈值到 < 4
- 忽略高分问题

## 支持

问题或建议？
1. 查看 `README.md` 详细文档
2. 查看 `SKILL.md` 完整流程
3. 调试：检查 todo list 和 agent 输出
