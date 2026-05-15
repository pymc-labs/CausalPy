# Bug Fix PR Review

Use this resource when a PR claims to correct broken behavior, regression, numerical error, documentation bug, CI failure, or user-reported issue.

## Review Focus

- Identify the bug being fixed and the evidence that it existed.
- Check that the root cause is addressed directly rather than masked by a broad fallback, silent exception handling, or special-case patch.
- Confirm the fix is as narrow as possible while preserving intended behavior for adjacent cases.
- Look for compatibility risk: released behavior should remain stable unless the previous behavior was clearly wrong and the change is documented.
- Verify error messages and project-specific exceptions remain helpful and consistent.
- For statistical or causal bugs, check the corrected estimand, contrast, uncertainty interval, indexing, or posterior calculation against the intended definition.

## Required Evidence

- There should be a regression test that fails on the old behavior and passes with the fix whenever practical.
- If the fix is for docs, notebooks, packaging, or CI, the relevant build/check should be run or the reason for not running it should be stated.
- Tests should cover the reported failure mode, not only a broader happy path.

## Request Changes When

- No test or reproducible evidence demonstrates the fixed bug.
- The patch hides the failure without addressing the underlying cause.
- The fix changes unrelated behavior or introduces a new public contract accidentally.
- The PR description, tests, and code disagree about what was broken.
- The fix relies on brittle assumptions about array shape, index order, sampling output, or file paths.
