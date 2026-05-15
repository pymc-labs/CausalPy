# Tests, CI, and Infrastructure PR Review

Use this resource when a PR changes tests, fixtures, pre-commit hooks, GitHub Actions, packaging, build configuration, dependencies, release tooling, or repository automation.

## Review Focus

- Confirm the change improves signal, reliability, speed, or maintainability rather than merely silencing failures.
- Check that CI changes do not weaken required coverage, skip important jobs, hide failures, or make checks pass only in a narrow environment.
- Review dependency changes for necessity, correct declaration location, version compatibility, and whether an existing dependency already solves the problem.
- Verify tests remain pytest-style, deterministic, quick, and located under `causalpy/tests/`.
- Check PyMC, PyTensor, and matplotlib test commands account for filesystem permissions and do not misdiagnose sandbox failures as code failures.
- Inspect fixtures and mocks for realism; avoid tests that mirror implementation details without asserting user-visible behavior.
- For packaging changes, check editable install, docs extras, generated `environment.yml` expectations, and import behavior.

## Required Evidence

- Infrastructure PRs should include before/after rationale or failure evidence for the changed automation.
- CI and hook changes should be exercised locally where feasible, usually through targeted `prek` or relevant build commands.
- Dependency or packaging changes should include import/install evidence or clear reasoning if not run locally.

## Review Output Emphasis

- In the PR summary, state which project safeguard or workflow changes and why it improves reliability, speed, or maintainability.
- In findings, foreground weakened checks, hidden failures, dependency risk, flaky tests, security-sensitive CI permissions, and packaging drift.
- In test evidence, distinguish local hook/test runs from remote CI status and note any jobs that were skipped or pending.
- In open questions, focus on required-check policy, dependency ownership, and whether the automation change should be narrower.

## Request Changes When

- A check is removed, skipped, or weakened without strong justification.
- Tests become slower, flaky, order-dependent, or overly coupled to internals.
- The PR edits generated environment files by hand instead of the source dependency configuration.
- CI secrets, tokens, permissions, or shell commands introduce unnecessary security risk.
- The change solves one branch's failure by making the default project safeguards less useful.
