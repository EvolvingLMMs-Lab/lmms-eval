# PR Code Review Skill

Simple, direct code review for GitHub PRs. Reviews diffs yourself using `gh` CLI - only spawns explore agents when you actually need codebase context.

## Usage

```
> Review PR #945
> Check all open PRs
```

## What it does

1. Fetches PR info and diff with `gh` commands
2. Reviews the changes directly (no agent swarm)
3. Checks against CLAUDE.md rules if the file exists
4. Posts a brief, human-readable comment

## What it checks

**Real bugs** - logic errors, missing returns, incorrect variable usage, runtime issues

**CLAUDE.md compliance** - type hints, docstrings, line length, PEP 8 naming (if project has CLAUDE.md)

## What it skips

- Pre-existing issues (lines not in diff)
- Linter/formatter stuff (CI handles this)
- Style nitpicks not in CLAUDE.md

## When agents are used

Only when you need codebase context you don't have. Most PRs don't need this - the diff tells the story.

Example: "I need to understand how error handling works in this module before I can judge if this change is correct" - then spawn an explore agent.

## Comment style

Writes like a colleague, not a checklist:

```markdown
### Code Review

Took a look at this PR. Found a couple things:

**Missing return statement** in `process_data` - computes the value but doesn't return it.
https://github.com/org/repo/blob/abc123/file.py#L42-L43

Otherwise looks good.
```

## Requirements

- `gh` CLI installed and authenticated
- Access to the repository
