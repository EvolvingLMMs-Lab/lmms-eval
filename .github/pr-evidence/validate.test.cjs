"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const test = require("node:test");
const { validateEvidence, run } = require("./validate.cjs");

// Synthetic reports test policy behavior; these are not claims of model runs.
const head = "0123456789abcdef0123456789abcdef01234567";
const command = "./.venv/bin/python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct --tasks mme --limit 8";
const report = [
  "### End-to-end validation",
  "- E2E status: `PASS`",
  "- Exact command:",
  "  ```bash",
  "  uv sync --locked",
  `  ${command}`,
  "  ```",
  "- Result: Completed 8 samples; mme_perception_score=100.",
  "- Evidence: Loaded 8 images, generated non-empty predictions; mme_perception_score=100.",
  "- Model/backend: qwen2_5_vl, Qwen/Qwen2.5-VL-3B-Instruct",
  "- Dataset split and sample size: mme test, N=8",
  "- Hardware: NVIDIA A100 80GB",
  "- [x] I verified this change end-to-end through `lmms_eval` with real data/media and a supported model backend.",
].join("\n");
const modelFile = { filename: "lmms_eval/models/chat/vllm_generate.py", status: "modified" };
const docsFile = { filename: "docs/guides/model_guide.md", status: "modified" };
const docsReport = report
  .replace(command, "node --test .github/pr-evidence/validate.test.cjs")
  .replace("`PASS`", "`NOT APPLICABLE`")
  .replace(/^- (?:Model\/backend|Dataset split and sample size|Hardware):.*\n?/gm, "")
  .replace(/^- \[x\].*$/m, "");
const check = (body = report, files = [modelFile], overrides = {}) =>
  validateEvidence({ body, head: { sha: head }, draft: false, ...overrides }, files);

test("the original E2E fields are sufficient for a modified existing model", () => {
  assert.deepEqual(check().errors, []);
  assert.equal(check().e2eRequired, true);
});

test("PR #1544 regression: prose-only validation of an existing model fails", () => {
  const result = check("## Validation\nTested on Intel BMG B70 (XPU). Runs to completion on nextqa_mc_test.\n- [x] Bug fix (non-breaking change)");
  assert.equal(result.e2eRequired, true);
  assert(result.errors.some((error) => error.includes("Exact command")));
  assert(result.errors.some((error) => error.includes("E2E status")));
});

for (const [filename, status, previous_filename] of [
  ["lmms_eval/models/chat/new_model.py", "added"],
  ["lmms_eval/tasks/task/config.yaml", "modified"],
  ["lmms_eval/tasks/task/utils.py", "modified"],
  ["lmms_eval/api/model.py", "modified"],
  ["lmms_eval/evaluator.py", "modified"],
  ["pyproject.toml", "modified"],
  ["uv.lock", "modified"],
  ["lmms_eval/models/simple/old.py", "removed"],
  ["docs/old-model.md", "renamed", "lmms_eval/models/simple/old.py"],
  ["new-runtime/package.py", "added"],
]) {
  test(`${status} ${filename} cannot use a docs exemption`, () => {
    const result = check(docsReport + "\n- [x] Documentation update", [{ filename, status, previous_filename }]);
    assert.equal(result.e2eRequired, true);
    assert(result.errors.some((error) => error.includes("E2E status")));
  });
}

for (const filename of [docsFile.filename, "AGENTS.md", ".github/workflows/ci.yml", ".github/pr-evidence/validate.cjs", "test/eval/test_example.py", ".gitignore"]) {
  test(`${filename} accepts documented non-runtime validation`, () => {
    const result = check(docsReport, [{ filename, status: "modified" }]);
    assert.equal(result.e2eRequired, false);
    assert.deepEqual(result.errors, []);
  });
}

test("docs can report checks under Validation without filling model E2E fields", () => {
  const body = "## Validation\n- node --test .github/pr-evidence/validate.test.cjs | result: pass\n- E2E status: `NOT APPLICABLE`";
  assert.deepEqual(check(body, [docsFile]).errors, []);
});

test("mixed docs/runtime files and declared integrations require E2E", () => {
  assert.equal(check(docsReport, [docsFile, modelFile]).e2eRequired, true);
  for (const kind of ["New benchmark/task", "New model integration"]) {
    assert.equal(check(docsReport + `\n- [x] ${kind}`, [docsFile]).e2eRequired, true);
  }
});

test("an empty file list cannot grant an exemption", () => {
  assert.equal(check(docsReport, []).e2eRequired, true);
});

test("blank fields cannot consume the following field's value", () => {
  for (const field of ["Result", "Evidence", "Model/backend", "Dataset split and sample size", "Hardware"]) {
    const body = report.split("\n").map((line) => line.startsWith(`- ${field}:`) ? `- ${field}:` : line).join("\n");
    assert(check(body).errors.length > 0, field);
  }
});

test("template placeholders, bare PASS, and comments are not evidence", () => {
  for (const value of ["N/A", "`TODO`", "TBD", "<paste logs here>", "PASS", "<!-- a completed run -->", "..."]) {
    assert(check(report.replace(/^- Evidence:.*$/m, `- Evidence: ${value}`)).errors.some((error) => /Evidence|logs/.test(error)), value);
  }
  assert(check(`<!--\n${report}\n-->`).errors.length > 0);
  assert(check(`\`\`\`\`markdown\n${report}\n\`\`\`\``).errors.length > 0);
});

test("the unfilled PR template fails for runtime changes", () => {
  const template = readFileSync(join(__dirname, "../pull_request_template.md"), "utf8");
  assert(check(template).errors.length > 0);
});

test("Windows line endings and field comments are supported", () => {
  const body = report.replace("- Hardware: NVIDIA A100 80GB", "- Hardware: NVIDIA A100 80GB <!-- GPU used -->").replace(/\n/g, "\r\n");
  assert.deepEqual(check(body).errors, []);
});

test("filling the existing PR template passes without additional fields", () => {
  const template = readFileSync(join(__dirname, "../pull_request_template.md"), "utf8");
  const body = template
    .replace("`PASS / NOT RUN / NOT APPLICABLE`", "`PASS`")
    .replace("# Paste the exact command here.", command)
    .replace("- Model/backend:", "- Model/backend: qwen2_5_vl, Qwen/Qwen2.5-VL-3B-Instruct")
    .replace("`N=`", "mme test, N=8")
    .replace("- Hardware:", "- Hardware: NVIDIA A100 80GB")
    .replace("- Result:", "- Result: Completed 8 samples; mme_perception_score=100.")
    .replace("- Evidence:", "- Evidence: Loaded 8 images; non-empty predictions; mme_perception_score=100.")
    .replace("- [ ] I verified", "- [x] I verified");
  assert.deepEqual(check(body).errors, []);
});

test("comment-only, placeholder, and missing commands fail", () => {
  for (const replacement of ["# python -m lmms_eval --tasks mme", "python -m lmms_eval --tasks <task>", "TODO", ""]) {
    const body = report.replace("  uv sync --locked\n", "").replace(command, replacement);
    assert(check(body).errors.some((error) => /command|CLI/.test(error)), replacement);
  }
});

test("helper tests alone cannot satisfy model E2E validation", () => {
  assert(check(report.replace(command, "./.venv/bin/python -m pytest test/models")).errors.some((error) => error.includes("CLI")));
});

test("public CLI launcher variants are accepted", () => {
  for (const launcher of ["python3.12 -m lmms_eval", "uv run python -m lmms_eval", "uv run lmms-eval", "./.venv/bin/lmms-eval"]) {
    assert.deepEqual(check(report.replace("./.venv/bin/python -m lmms_eval", launcher)).errors, [], launcher);
  }
});

test("positive sample count and an explicit attestation are required", () => {
  for (const replacement of ["N=0", "N=-1", "N="]) {
    assert(check(report.replace("N=8", replacement)).errors.some((error) => error.includes("positive sample count")));
  }
  assert(check(report.replace("- [x] I verified", "- [ ] I verified")).errors.some((error) => error.includes("attestation")));
});

test("blocked runs must stay in draft", () => {
  const body = report.replace("`PASS`", "`NOT RUN`");
  assert(check(body).errors.length > 0);
  assert.deepEqual(check(body, [modelFile], { draft: true }).errors, []);
  assert(check(null, [], { draft: true }).draft);
});

async function runWithApi(pull, files, eventBody = report) {
  const failures = [];
  const notices = [];
  let listed = false;
  const context = { repo: { owner: "example", repo: "lmms-eval" }, payload: { pull_request: { number: 1544, body: eventBody } } };
  const listFiles = () => {};
  const github = {
    rest: { pulls: { get: async (params) => {
      assert.equal(params.pull_number, 1544);
      return { data: pull };
    }, listFiles } },
    paginate: async (method, params) => {
      assert.equal(method, listFiles);
      assert.equal(params.per_page, 100);
      listed = true;
      return files;
    },
  };
  await run({ github, context, core: { setFailed: (message) => failures.push(message), notice: (message) => notices.push(message) } });
  return { failures, notices, listed };
}

test("reruns use current PR metadata instead of stale successful event evidence", async () => {
  const outcome = await runWithApi({ body: "", head: { sha: head }, changed_files: 1 }, [modelFile]);
  assert.equal(outcome.failures.length, 1);
  const fixed = await runWithApi({ body: report, head: { sha: head }, changed_files: 1 }, [modelFile], "");
  assert.deepEqual(fixed.failures, []);
});

test("partial API file lists fail closed", async () => {
  const outcome = await runWithApi({ body: docsReport, head: { sha: head }, changed_files: 2 }, [docsFile]);
  assert(outcome.failures[0].includes("every changed file"));
});

test("drafts do not need a file listing", async () => {
  const outcome = await runWithApi({ draft: true, body: null, head: { sha: head } }, []);
  assert.deepEqual(outcome.failures, []);
  assert.equal(outcome.listed, false);
  assert(outcome.notices[0].includes("Draft"));
});
