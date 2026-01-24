---
name: pr-code-review
description: Review GitHub PRs for bugs and code quality. Use when user says "review PR", "check PR #123", or mentions PR review.
---

# PR Code Review

Straightforward code review for GitHub pull requests. Reviews the diff directly, only spawns agents when you need deeper codebase context.

## Triggers

- "review PR #123"
- "check this PR"
- "code review"

## How it works

You review the PR yourself using `gh` CLI. Only use explore agent if you need to understand unfamiliar parts of the codebase.

## Review process

### 1. Check if PR is reviewable

```bash
gh pr view {number} --json state,isDraft,title,body
```

Skip if closed, draft, or already has your review:
```bash
gh api repos/{owner}/{repo}/pulls/{number}/comments | grep -c "Claude Code"
```

### 2. Get the diff and context

```bash
gh pr view {number}
gh pr diff {number}
```

Read CLAUDE.md if it exists - it contains project coding standards.

### 3. Review the changes

Look at the diff yourself. Focus on:

**Real bugs** - logic errors, missing error handling, runtime issues, incorrect variable usage

**CLAUDE.md violations** - if the project has one, check for type hints, docstrings, line length (88), PEP 8 naming, uv-only (no pip)

**Skip these** - pre-existing issues (lines not in diff), linter-catchable stuff, style nitpicks not in CLAUDE.md

### 4. Use explore agent only when needed

If you're unfamiliar with how the codebase does something, spawn an explore agent:

```
Task(subagent_type="explore", prompt="Find how error handling is done in lmms_eval/models/ - what patterns exist?")
```

Don't spawn agents for every PR. Most diffs are self-contained.

### 5. Post your review

Get the commit SHA for links:
```bash
gh pr view {number} --json headRefOid --jq '.headRefOid'
```

Post with:
```bash
gh pr comment {number} --body "..."
```

## Comment format

Write like a helpful colleague, not a robot. Keep it brief and actionable.

**When you find issues:**

```markdown
### Code Review

Took a look at this PR. Found a couple things worth addressing:

**Missing return statement** in `ovo_doc_to_target` - the function computes the value but never returns it.
https://github.com/{owner}/{repo}/blob/{SHA}/{file}#L102-L103

**No docstrings on public functions** - CLAUDE.md requires these for public APIs.
https://github.com/{owner}/{repo}/blob/{SHA}/{file}#L17-L45

Otherwise looks good. Let me know if you have questions.
```

**When it looks good:**

```markdown
### Code Review

Reviewed the changes - looks good to me. Clean implementation, follows existing patterns.
```

**When skipping:**

Just tell the user why: "PR #123 is a draft, skipping review."

## What NOT to flag

- Issues on lines you didn't modify (pre-existing)
- Formatting/linting issues (CI catches these)
- Personal style preferences not in CLAUDE.md
- Lines with `# noqa` or similar ignore comments

## Link format

Use full SHA, not short:
```
https://github.com/{owner}/{repo}/blob/{FULL_SHA}/{file_path}#L{start}-L{end}
```

## Multiple PRs

```
> Review all open PRs
```

Get the list and review each:
```bash
gh pr list --state open --json number
```
