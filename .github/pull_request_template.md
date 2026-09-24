## Summary
<!-- Max 3 bullets. -->
- 
- 
- 

## In scope
<!-- Explicitly list what this PR changes. -->
- 

## Out of scope
<!-- Explicitly list what this PR does NOT change. -->
- 

## Validation
<!--
List static checks such as lint, compilation, and unit tests here.
Use this format (max 3 bullets):
`<command>` | key checks: `<...>` | result: `pass/fail`
-->
- 

### End-to-end validation
<!--
Required for changes to evaluation behavior, including fixes/refactors of
existing models/tasks, shared evaluation code, dependencies, and new
integrations, before the PR is marked ready for review. Run through the public
lmms_eval CLI using a real dataset sample, real media, and an actual supported
model/backend. Docs/tests/CI-only PRs may use NOT APPLICABLE and list their
checks in Validation above.

Mock-only tests, helper-function tests, compilation, lint, and temporary
uncommitted scripts do not count as end-to-end validation. If the run cannot
be completed yet, leave E2E status as NOT RUN and keep the PR in draft.
-->
- E2E status: `PASS / NOT RUN / NOT APPLICABLE`
- Exact command:
  ```bash
  # Paste the exact command here.
  ```
- Model/backend:
- Dataset split and sample size: `N=`
- Hardware:
- Result:
- Evidence: <!-- Link a run log/artifact, or paste a concise excerpt showing dataset loading, media resolution, inference, a non-empty prediction, and the emitted metric. -->

- [ ] I verified this change end-to-end through `lmms_eval` with real data/media and a supported model backend.

## Risk / Compatibility
<!-- 1-2 bullets. Note breaking changes, behavior changes, or migration impact. -->
- 

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature
- [ ] New benchmark/task
- [ ] New model integration
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
