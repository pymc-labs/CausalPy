# PyMC Migration Baseline Harness

This permanent harness produces reproducible evidence for the PyMC 5 → PyMC 6 migration at the only two revisions that may be attributed to that migration: PyMC 5 reference `79c0a87072fd4653bfaed1eb085f965594c7f03a` and PyMC 6 migration candidate `18a524a1a8512aaa21c46e0ccddbc54501c9eb1a`. It rejects every other source revision so later features are investigated as separate changes rather than mislabeled migration drift.

The tracked implementation is `scripts/migration_baseline/harness.py`; generated JSON and Markdown evidence belongs outside every Git checkout. The harness rejects destinations inside its own checkout or either sampled checkout, and creates evidence files without replacing an existing path.

## Historical v1 result and v2 evidence requirement

The prior schema-v1 coordinator run passed all registered gates. Its issue comment is [#1048 evidence comment](https://github.com/pymc-labs/CausalPy/issues/1048#issuecomment-5116536256), and its immutable evidence manifest is [gist revision `ac7db2676caf0eec0ae2da46ac52e48b3b00f86a`](https://gist.github.com/cetagostini/62d0ebf197c99fd7eef4336fb7de46a1/ac7db2676caf0eec0ae2da46ac52e48b3b00f86a).

Schema v2 intentionally rejects those v1 artifacts: it adds executing-harness binding, source-manifest validation, imported-runtime provenance, and fresh-batch identity. The coordinator must generate a new v2 evidence directory and immutable attachment before this protocol is considered satisfied. Do not overwrite, append to, or relabel the v1 directory or gist as v2 evidence.

## What is captured

The fixed suite has two representative Bayesian experiments using input rows serialized directly in the harness rather than regenerated through package-version-dependent simulation code:

- `DifferenceInDifferences` with `LinearRegression`, including its public effect-summary table, treatment-effect posterior, fitted draw-wise R², and treated-post conditional-mean counterfactual.
- `SyntheticControl` with `WeightedSumFitter`, including its public average/cumulative effect-summary rows, post-impact posteriors, fitted draw-wise R², and post-treatment conditional-mean counterfactual.

The embedded v2 manifest fixes every fixture row and hash, public table schema and bindings, series name, canonical dimensions, shape, coordinate values, selector, and metric cardinality. A self-consistent but incomplete or substituted artifact is not evidence.

The DiD fixture includes stable `unit` labels required by its public constructor. They are omitted from the model formula and do not alter the serialized outcomes or estimand. Its public scalar effect is canonicalized to `(chain, draw)` when a backend exposes only a singleton `treated_units` dimension. A non-singleton or otherwise unexpected effect dimension fails capture.

All counterfactuals are adapter-returned `mu` conditional expected values. The harness rejects noisy `y_hat` predictions, missing canonical dimensions, changed coordinate values, and metric selectors outside the fixed manifest.

## Registered capture and runtime protocol

- Explicit HDI probability: `0.94`, passed to every `effect_summary()` as `alpha=0.06`.
- Sampler: PyMC NUTS, four chains with `cores=1`, 1,000 tuning iterations, 1,000 retained draws, master seed `1048`, target acceptance `0.95`, and maximum tree depth `12`.
- `cores=1` is intentional on every platform. In particular it is mandatory on local macOS to avoid Accelerate/numba fork failures and to make the chain schedule deterministic.
- Capture fails before comparison for divergent chains, tree-depth saturation when that statistic is exposed, missing sample statistics, non-finite values, rank R-hat above `1.01`, bulk ESS below `400`, or tail ESS below `400`.
- Tail ESS uses the explicit legacy probability pair `(0.05, 0.95)` in both environments; an ArviZ API that cannot accept that registered policy invalidates capture.
- The draw-wise R² formula is the CausalPy formula evaluated independently for every `(chain, draw, treated_unit)`: `var_obs(mu) / (var_obs(mu) + var_obs(y - mu))` with `ddof=0`. The harness keeps posterior draws long enough to calculate MCSE and convergence diagnostics before serializing only summaries.

Each sampled checkout and the harness checkout must have an empty `git status --porcelain`. The harness verifies that its executing file SHA-256 matches the tracked Git blob at its recorded commit. Every sampled artifact then records that harness SHA-256, blob SHA-256, and commit; `compare` requires all four to match the currently executing harness.

Capture imports CausalPy only after pinning the selected checkout. It also verifies the imported runtime before sampling:

- `pymc5` requires imported PyMC major `5`, PyTensor major `2`, and ArviZ major `0`.
- `pymc6` requires imported PyMC major `6`, PyTensor major `3`, and ArviZ major `1`.
- The imported dependency paths must be below the active `sys.prefix`; `sys.executable`, `sys.prefix`, imported versions and module paths are recorded.
- The active CausalPy distribution must be an editable install whose `direct_url.json` target is exactly `--repo-root`.

The two repeat captures for one stack must have identical runtime provenance, posterior summaries, and sampling-quality evidence as well as exact same-stack raw-draw digests. The two stacks must use distinct prefixes while matching platform, machine, Python version/implementation, and NumPy/pandas/xarray versions; otherwise a host or shared-runtime change is not treated as migration evidence. PyMC, PyTensor, and ArviZ are intentionally stack-specific.

## v2 coordinator reproducibility procedure

Run the four capture commands as independent processes from a clean, committed harness checkout. One coordinator-generated canonical UUID is required for the whole batch; each capture receives its fixed role and a fresh capture UUID is generated inside the harness. The outputs below are create-only: use a newly created evidence directory, not an existing directory or old v1 evidence.

The coordinator must provision `PYMC6_ROOT` as a separate clean detached worktree at `18a524a1a8512aaa21c46e0ccddbc54501c9eb1a` and install it into its own editable-install prefix. Do not use the committed `migration/1048-baseline-harness` checkout as `PYMC6_ROOT`: its source `HEAD` intentionally differs from the migration candidate.

Set the six coordinator variables below to your own locations. `WORKTREES` is any
directory outside every CausalPy checkout; `MAMBA` is whichever environment
manager provides the two prefixes.

```bash
set -euo pipefail

MAMBA="$(command -v mamba)"
WORKTREES="${WORKTREES:?set to a directory outside every CausalPy checkout}"
MIGRATION_ROOT="$WORKTREES/CausalPy-1048-baselines"
PYMC5_ROOT="$WORKTREES/CausalPy-1048-pymc5"
PYMC5_PREFIX="$WORKTREES/.mamba/CausalPy-1048-pymc5"
PYMC6_ROOT="$WORKTREES/CausalPy-1048-pymc6"
PYMC6_PREFIX="$WORKTREES/.mamba/CausalPy-1048-pymc6"
BATCH_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
EVIDENCE_ROOT="$WORKTREES/migration-baseline-v2-${BATCH_ID}"
HARNESS="$MIGRATION_ROOT/scripts/migration_baseline/harness.py"

mkdir "$EVIDENCE_ROOT"

"$MAMBA" run -p "$PYMC5_PREFIX" python "$HARNESS" capture \
  --stack pymc5 --capture-role reference_first --batch-id "$BATCH_ID" \
  --repo-root "$PYMC5_ROOT" --output "$EVIDENCE_ROOT/pymc5-run-1.json"
"$MAMBA" run -p "$PYMC5_PREFIX" python "$HARNESS" capture \
  --stack pymc5 --capture-role reference_second --batch-id "$BATCH_ID" \
  --repo-root "$PYMC5_ROOT" --output "$EVIDENCE_ROOT/pymc5-run-2.json"
"$MAMBA" run -p "$PYMC6_PREFIX" python "$HARNESS" capture \
  --stack pymc6 --capture-role candidate_first --batch-id "$BATCH_ID" \
  --repo-root "$PYMC6_ROOT" --output "$EVIDENCE_ROOT/pymc6-run-1.json"
"$MAMBA" run -p "$PYMC6_PREFIX" python "$HARNESS" capture \
  --stack pymc6 --capture-role candidate_second --batch-id "$BATCH_ID" \
  --repo-root "$PYMC6_ROOT" --output "$EVIDENCE_ROOT/pymc6-run-2.json"
"$MAMBA" run -p "$PYMC6_PREFIX" python "$HARNESS" compare \
  --reference-first "$EVIDENCE_ROOT/pymc5-run-1.json" \
  --reference-second "$EVIDENCE_ROOT/pymc5-run-2.json" \
  --candidate-first "$EVIDENCE_ROOT/pymc6-run-1.json" \
  --candidate-second "$EVIDENCE_ROOT/pymc6-run-2.json" \
  --output "$EVIDENCE_ROOT/1048-baseline-comparison.json" \
  --report "$EVIDENCE_ROOT/1048-baseline-report.md"
```

The comparator requires four distinct paths, exact role order, one shared batch UUID, and four distinct capture UUIDs. It reads each JSON input once, hashes that exact byte buffer, and carries the buffer-derived hash into the report. It verifies exact raw-draw digests, posterior summaries, and sampling-quality evidence only within each stack; a mismatch makes the entire comparison fail as non-deterministic evidence. It never compares a PyMC 5 digest or raw draw with a PyMC 6 digest or raw draw.

A failed numerical comparison writes its fresh JSON decision and Markdown report, then exits with status `1`; malformed or invalid evidence exits with status `2`. The generated report records all four artifact paths and byte hashes, role/batch identity, clean checkout result, comparator identity, imported runtime provenance, and actual finite/convergence diagnostics. Attach it and its four input artifacts to #1048 with the command log. The static attachment outline is in [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md).

## Continuous verification of the capture path

`causalpy/tests/test_migration_baseline_harness.py` runs both fixed scenarios end
to end against the installed stack on every CI run, at a reduced posterior size
with relaxed convergence thresholds, and asserts that the captured fixtures,
effect-table bindings, series semantics, metric selectors and sampling-quality
evidence reproduce the manifest exactly. That reduced protocol is never
serialized as evidence: `REGISTERED_SAMPLING` is the only protocol `capture`
uses and the only one `compare` validates against. Its purpose is to fail fast
when a CausalPy or ArviZ API the capture path depends on drifts, instead of
after hours of coordinator sampling.

## Registered migration decision gates

The comparator applies these migration hard gates to every independent posterior scalar, including effect quantities, each draw-wise R² series, and each counterfactual coordinate:

1. `abs(candidate_mean - reference_mean) <= max(4 * hypot(reference_mcse, candidate_mcse), 1e-6 + 1e-4 * abs(reference_mean))`.
2. `abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1`, where `pooled_posterior_sd = sqrt((reference_sd² + candidate_sd²) / 2)`. A degenerate posterior uses the absolute gate because standardized drift has no meaningful denominator.
3. Effect-table schema, explicit 0.94 HDI column labels, metric selectors, dimension order, shape, and coordinate values must match exactly.

Mutual mean-in-94%-HDI containment is a diagnostic in the rendered table only. It does not pass or fail a migration gate.

## Reuse for #157

The harness is reusable infrastructure, not a claim that a PyMC 5 posterior is ground truth. #157 correctness tests can reuse its fixed fixture serialization, explicit HDI extraction, draw-wise R² calculation, semantic schema checks, and reporting layout, but its primary assertions must compare simulated-data estimates against known data-generating parameters with separately registered posterior-SD-unit tolerances. Single-realization 94% HDI coverage remains diagnostic in that later correctness work.
