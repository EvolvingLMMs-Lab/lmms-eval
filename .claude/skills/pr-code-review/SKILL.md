---
name: pr-code-review
description: Comprehensive code review for GitHub pull requests using parallel agents. Checks bugs, CLAUDE.md compliance, git history, previous PR patterns, and code comments. Use when reviewing PRs, doing code review, or when user mentions "review PR", "check PR", or provides a PR number.
---

# PR Code Review

Performs comprehensive, multi-dimensional code review for GitHub pull requests using 5 parallel Sonnet agents and confidence-based issue scoring.

## When to use

- User asks to review a specific PR (e.g., "review PR #123")
- User mentions "code review", "check this PR", "review pull request"
- User provides a PR number or URL
- User says "review all PRs" or "check open PRs"

## Quick start

```
User: "Review PR #945"
Assistant: [Launches this skill with PR number 945]
```

## Review process

This skill implements an 8-step comprehensive review:

1. **Eligibility check** (Haiku agent)
   - Verify PR is open, not draft, not already reviewed
   - Skip if ineligible

2. **Find CLAUDE.md files** (Haiku agent)
   - Locate root and directory-specific CLAUDE.md files
   - Use for compliance checking

3. **Get PR summary** (Haiku agent)
   - Extract PR description, changes, affected files
   - Understand the context

4. **Launch 5 parallel review agents** (Sonnet agents)
   - Agent #1: CLAUDE.md compliance
   - Agent #2: Shallow bug scan
   - Agent #3: Git history analysis
   - Agent #4: Previous PR patterns
   - Agent #5: Code comment compliance

5. **Score all issues** (parallel Haiku agents)
   - Each issue gets 0-100 confidence score
   - Use standardized rubric
   - Verify CLAUDE.md citations

6. **Filter issues**
   - Keep only issues with score >= 80
   - Skip review if no high-confidence issues

7. **Re-verify eligibility** (Haiku agent)
   - Ensure PR is still open and not already reviewed

8. **Post review comment** (gh command)
   - Format with issue descriptions
   - Include file links with full SHA
   - Add feedback footer

## Instructions

### Step 1: Create todo list

```
Use TodoWrite to create tracking list:
- Check PR eligibility
- Find CLAUDE.md files
- Get PR summary
- Launch 5 review agents
- Score issues
- Filter and verify
- Post review comment
```

### Step 2: Eligibility check

Launch Haiku agent to verify:
```
- Is PR closed? → SKIP
- Is PR draft? → SKIP
- Doesn't need review (automated/trivial)? → SKIP
- Already has Claude Code review? → SKIP

Use: gh pr view {number}
     gh api repos/{owner}/{repo}/pulls/{number}/comments
```

### Step 3: Find CLAUDE.md files

Launch Haiku agent to locate:
```
- Root CLAUDE.md
- Directory-specific CLAUDE.md files for modified paths

Use: gh pr view {number} --json files
     Check for CLAUDE.md in each directory
```

Return list of CLAUDE.md file paths (not contents).

### Step 4: Get PR summary

Launch Haiku agent to extract:
```
- What the PR adds/changes/fixes
- Key files modified
- Main functionality changes

Use: gh pr view {number}
     gh pr diff {number}
```

### Step 5: Launch 5 parallel review agents

**IMPORTANT**: Launch all 5 agents in parallel (single message, multiple Task calls).

#### Agent #1: CLAUDE.md Compliance (Sonnet)
```
Review PR for CLAUDE.md compliance:
- Read CLAUDE.md files
- Check code style (type hints, docstrings, line length 88, PEP 8)
- Check testing requirements
- Check package management (uv only, no pip)
- Note: CLAUDE.md is guidance for writing code, not all applies to review

Return issues with:
- Issue description
- CLAUDE.md rule violated
- File and line reference
```

#### Agent #2: Shallow Bug Scan (Sonnet)
```
Review PR for obvious bugs:
- Use gh pr diff {number}
- Focus ONLY on changed lines
- Look for: logic errors, incorrect variable usage, missing critical error handling, runtime errors
- Avoid: nitpicks, style issues, false positives, pre-existing issues

Return issues with:
- Issue description
- Bug type
- File and line reference
```

#### Agent #3: Git History Analysis (Sonnet)
```
Review PR against git history:
- Get modified files
- Use git blame and git log
- Look for: violated patterns, repeated historical bugs, relevant commit context

Return issues with:
- Issue description
- Historical context
- File and line reference
```

#### Agent #4: Previous PR Patterns (Sonnet)
```
Review PR against previous PR patterns:
- Get modified files
- Search for recent PRs touching same files
- Check comments on those PRs for applicable feedback

Use: gh pr list --search "path:{file}" --state merged

Return issues with:
- Issue description
- Pattern from PR #X
- File and line reference
```

#### Agent #5: Code Comment Compliance (Sonnet)
```
Review PR for code comment compliance:
- Read modified files
- Check for TODO comments, warnings, documented patterns
- Verify changes follow comment guidance

Return issues with:
- Issue description
- Comment guidance violated
- File and line reference
```

### Step 6: Score all issues

For EACH issue found by the 5 agents, launch a parallel Haiku agent to score it.

**Scoring rubric** (give to agent verbatim):
```
Score on 0-5 scale:
- 0: False positive. Not a real issue or pre-existing.
- 1: Low confidence. Might be real but likely false positive.
- 2: Uncertain. Could be real but might be nitpick or minor.
- 3: Moderate confidence. Real issue but not critical.
- 4: High confidence. Real issue that will impact functionality.
- 5: Critical. Definitely a real bug that will cause problems.
```

**Agent instructions**:
```
Score this issue on 0-5 scale using the rubric above.

Issue: [issue description]

For CLAUDE.md issues: Verify CLAUDE.md actually says this.
For bugs: Verify with gh pr diff that this is real.
Check for false positives: pre-existing issues, lines not modified, linter-catchable.

Return: score (0-5) and 1-2 sentence explanation.
```

### Step 7: Filter and re-verify

```
1. Filter issues: Keep only score >= 4 (high confidence or critical)
2. If no issues >= 4: STOP, do not post review
3. Re-verify eligibility with Haiku agent (same as Step 2)
4. If not eligible: STOP
```

### Step 8: Post review comment

Get PR head commit SHA:
```bash
gh pr view {number} --json headRefOid --jq '.headRefOid'
```

Format comment:
```markdown
### Code review

Found {N} issues:

1. {brief description} ({CLAUDE.md says "..."} or {bug due to ...})

{link with full SHA}

2. ...

🤖 Generated with [Claude Code](https://claude.ai/code)

<sub>- If this code review was useful, please react with 👍. Otherwise, react with 👎.</sub>
```

Or if no issues:
```markdown
### Code review

No issues found. Checked for bugs and CLAUDE.md compliance.

🤖 Generated with [Claude Code](https://claude.ai/code)
```

**Link format**:
```
https://github.com/{owner}/{repo}/blob/{FULL_SHA}/{file_path}#L{start}-L{end}

Requirements:
- Use FULL commit SHA (not short)
- # before line numbers
- L prefix for lines
- Range format: L{start}-L{end}
- Include 1+ lines of context around issue
```

Post with:
```bash
gh pr comment {number} --body "{formatted_comment}"
```

## False positives to avoid

**Do NOT flag**:
- Pre-existing issues (on lines not modified in PR)
- Linter/typechecker catchable issues (will run in CI)
- Pedantic nitpicks
- Style issues not in CLAUDE.md
- Issues silenced with lint ignore comments
- Intentional functionality changes
- General code quality (unless CLAUDE.md requires it)

## Examples

### Example 1: Review single PR
```
User: Review PR #945
Assistant: [Creates todo list, runs 8-step process, posts review]
```

### Example 2: Review multiple PRs
```
User: Review all open PRs
Assistant: [Gets list with gh pr list, reviews each in sequence]
```

### Example 3: PR is not ready
```
User: Review PR #959
Assistant: [Checks eligibility, finds incomplete description, skips with explanation]
```

## Best practices

1. **Always use parallel agents** where possible (Steps 5 and 6)
2. **Trust agent scores** - they do deep analysis
3. **Skip low-confidence issues** - only report score >= 80
4. **Verify eligibility twice** - before and after scoring
5. **Use full SHA in links** - required for GitHub rendering
6. **Keep comments brief** - focus on high-impact issues
7. **Update todos frequently** - show progress to user

## Common issues

**Agents give contradictory scores**: Trust the explanation, not just the number. If explanation says "not a problem" but score is 75, treat as false positive (score should be 0).

**Too many issues found**: This is good! The parallel agent approach catches more. The scoring step filters to only high-confidence issues (score >= 4).

**PR changed during review**: The re-verification step (Step 7) catches this.

**No CLAUDE.md files**: Review still works, just skips CLAUDE.md compliance checks.

## Requirements

- GitHub CLI (`gh`) installed and authenticated
- Access to repository
- Ability to post comments on PRs

## Tool usage

- **Task tool**: Launch all Haiku and Sonnet agents
- **TodoWrite**: Track progress through 8 steps
- **Bash (gh)**: Interact with GitHub (no WebFetch)
- **Read**: Access CLAUDE.md files if needed

## Output format

After completing review:

```
✓ Reviewed PR #{number}
- Found {N} high-confidence issues (score >= 4)
- Posted review comment: {URL}
```

Or:
```
✓ Reviewed PR #{number}
- No issues found
- Posted approval comment: {URL}
```

Or:
```
⊘ Skipped PR #{number}
- Reason: {eligibility issue}
```
