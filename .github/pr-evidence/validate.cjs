"use strict";

// Only docs, tests, and repository maintenance can omit model E2E evidence.
// Unknown paths default to requiring it. Check both sides of renames below.
function isNonRuntimePath(filename) {
  return /^(?:docs\/|test\/|tests\/|\.github\/)/.test(filename) ||
    /\.(?:md|rst)$/i.test(filename) ||
    /^(?:LICENSE|CITATION\.cff|\.gitignore|\.pre-commit-config\.yaml)$/.test(filename);
}

function requiresE2E(files, body) {
  const declaresIntegration = /^[ \t]*-[ \t]*\[[xX]\][ \t]*New (?:benchmark\/task|model integration)[ \t]*$/m.test(body);
  return declaresIntegration || files.length === 0 || files.some((file) =>
    [file.filename, file.previous_filename].filter(Boolean).some((path) => !isNonRuntimePath(path))
  );
}

function isFilled(value) {
  const text = value.replace(/`/g, "").trim();
  return text.length >= 3 &&
    !/^(?:n\/?a|not run|not applicable|none|todo|tbd|pending|pass|fail|\.\.\.)[.!]?$/i.test(text) &&
    !/<[^>]+>|\b(?:TODO|TBD)\b/.test(text);
}

function validateEvidence(pull, files) {
  if (pull.draft) return { errors: [], draft: true };

  // Template comments and sample field names inside code blocks are not evidence.
  const body = (pull.body || "").replace(/<!--[\s\S]*?(?:-->|$)/g, "").replace(/\r\n/g, "\n");
  const fieldsBody = body.replace(/^[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$/gm, "");
  const e2eRequired = requiresE2E(files, fieldsBody);
  if (!e2eRequired) return { errors: [], draft: false, e2eRequired };

  const readField = (label) => {
    const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Horizontal whitespace only: an empty field must not swallow the next line.
    return fieldsBody.match(new RegExp(`^[ \\t]*-[ \\t]*${escaped}:[ \\t]*([^\\n]*)$`, "im"))?.[1]?.trim() || "";
  };
  const errors = [];
  const requireField = (label, message) => {
    const value = readField(label);
    if (!isFilled(value)) errors.push(message);
    return value;
  };

  requireField("Result", "Report the observed result after the change in `Result`.");
  const evidence = requireField("Evidence", "Link logs/artifacts or provide concrete output in `Evidence`.");
  if (isFilled(evidence) && evidence.length < 10) errors.push("Provide a concrete output excerpt or artifact link in `Evidence`.");

  const command = body.match(
    /^[ \t]*-[ \t]*Exact command:[ \t]*\n[ \t]*```(?:bash|sh|shell)?[ \t]*\n([\s\S]*?)^[ \t]*```[ \t]*$/im,
  )?.[1]?.trim() || "";
  const executableCommand = command.split("\n").filter((line) => line.trim() && !line.trim().startsWith("#")).join("\n");
  if (!executableCommand || /<[^>]+>|\b(?:TODO|TBD)\b|^[ \t]*\.\.\.[ \t]*$/m.test(executableCommand)) {
    errors.push("Provide the exact runnable command without placeholders in the fenced `Exact command` block.");
  }

  const status = readField("E2E status").replace(/`/g, "").toUpperCase();
  if (status !== "PASS") {
    errors.push("This PR changes runtime files or declares an integration. Complete real E2E validation and set `E2E status` to `PASS`; otherwise keep it in draft.");
  }
  if (!/(?:\bpython(?:3(?:\.\d+)?)?\s+-m\s+lmms_eval\b|(?:^|\s|\/)lmms-eval(?=\s|$))/m.test(executableCommand)) {
    errors.push("Include the exact public `lmms_eval` CLI invocation in `Exact command` for E2E validation.");
  }
  requireField("Model/backend", "Identify the real model/backend used for the run.");
  const dataset = requireField("Dataset split and sample size", "Identify the dataset, split, and sample size.");
  if (!/\bN[ \t]*=[ \t]*[1-9]\d*\b/i.test(dataset)) errors.push("Provide a positive sample count as `N=<number>`.");
  requireField("Hardware", "Identify the hardware or hosted inference service used.");
  if (!/^[ \t]*-[ \t]*\[[xX]\][ \t]*I verified this change end-to-end through `lmms_eval` with real data\/media and a supported model backend\.[ \t]*$/m.test(fieldsBody)) {
    errors.push("Check the end-to-end verification attestation after completing the run.");
  }
  return { errors, draft: false, e2eRequired };
}

async function run({ github, context, core }) {
  const params = { owner: context.repo.owner, repo: context.repo.repo, pull_number: context.payload.pull_request.number };
  // A rerun must inspect the current description and head, not an old event body.
  const { data: pull } = await github.rest.pulls.get(params);
  const files = pull.draft ? [] : await github.paginate(github.rest.pulls.listFiles, { ...params, per_page: 100 });
  // GitHub caps this endpoint at 3,000 files. Never grant an exemption on a partial list.
  if (!pull.draft && files.length !== pull.changed_files) {
    core.setFailed("Could not inspect every changed file. Reduce the PR size before validating reproduction evidence.");
    return;
  }
  const { errors, draft, e2eRequired } = validateEvidence(pull, files);
  if (draft) {
    core.notice("Draft PR: reproduction evidence is required before marking it ready for review.");
  } else if (errors.length) {
    core.setFailed(`Missing required reproduction evidence (see CONTRIBUTING.md and the PR template):\n- ${errors.join("\n- ")}`);
  } else if (!e2eRequired) {
    core.notice("Docs/tests/CI-only PR: model E2E evidence is not required. Report relevant checks under Validation.");
  } else {
    core.notice("Required reproduction evidence fields are present. Reviewers must verify the commands and results; this check does not run model inference.");
  }
}

module.exports = { validateEvidence, run };
