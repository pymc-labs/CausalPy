# What to Look For

Severity-sorted patterns. This is the diagnostic checklist — walk it explicitly during step 3 of the workflow.

The patterns below are organised by severity, then by recurring **shape**. Each shape has a short rationale and a concrete example drawn from past reviews where possible. When a pattern requires deep CausalPy knowledge, it's been split into [code-patterns.md](code-patterns.md) (Python source) or [docs-patterns.md](docs-patterns.md) (notebooks + docs).

## Must-fix patterns

These are bugs or contract violations. Posting as `must-fix` should mean "I will block approval until this is addressed."

### MF-1: Subclass override drops base-class behaviour

A subclass overrides a method but doesn't forward all the kwargs the base class forwards.

- **Canonical example**: `BayesianBasisExpansionTimeSeries._clone()` and `StateSpaceTimeSeries._clone()` in PR #826 omitted `priors=self._user_priors`. Consequence: user priors silently dropped on clone.
- **How to spot**: when a diff adds a method override, read the base-class version side-by-side. Diff their kwargs.
- See [code-patterns.md § _clone() pattern](code-patterns.md#clone-pattern) for the canonical CausalPy `_clone()` shape.

### MF-2: Constrained string parameter not typed as `Literal`

`AGENTS.md` is explicit: "use `Literal` for constrained string parameters."

- **Canonical example**: `selection_method: str` accepting `"sequential"` or `"random"` should be `Literal["sequential", "random"]`.
- **How to spot**: any new function/method parameter with a type of `str` whose docstring or implementation enumerates a fixed set of accepted values.
- This is also a candidate for a Ruff/mypy custom rule eventually — file an issue if the same instance appears across multiple PRs.

### MF-3: Behavioural change to existing function with no test that pins the new behaviour

Common after bug fixes. The fix changes math/logic, but the only tests are pre-existing tests that happen to still pass — they don't *require* the new behaviour.

- **Canonical example**: PR #826 changed the assurance simulation formula. Round-1 review flagged the missing test; round-2 confirmed two new tests now pin the corrected math.
- **How to spot**: look for diff hunks that change a constant, formula, or branch condition in non-test code with no corresponding new assertion in tests.
- The verification ask is specific: a test that would fail under the old code and pass under the new.

### MF-4: New public API not consistent with sibling implementations

If `BaseExperiment` subclasses all expose `supports_ols` and `supports_bayes`, a new subclass missing them is a bug. Same for `PyMCModel` subclasses, `CheckResult`-producing checks, etc.

- See [code-patterns.md § BaseExperiment contract](code-patterns.md#baseexperiment-contract) and [§ PyMCModel contract](code-patterns.md#pymcmodel-contract).

## Should-fix patterns

Design judgement, not bugs. Block on these only if the contributor pushes back without good reason.

### SF-1: Default value of a memory-heavy retainer is "on"

Result/check classes that retain the full fitted model (including `InferenceData`) by default impose a hidden cost on every user, even those who only need summary stats.

- **Canonical example**: PR #826 `FalsificationResult` retains the full `BaseExperiment`. Should default to summary-only with `store_experiments=True` as opt-in.
- **How to spot**: any new dataclass/class with a field that holds a fitted experiment, posterior, or large array, instantiated by default during `run()`/`fit()`.
- If the contributor pushes back: ask whether common-case users will pay for the inspection feature they don't use.

### SF-2: Bare `except Exception` (or `except:`) in library code

Hides unexpected bugs as well as expected failures. Acceptable when the goal is "best effort and log everything", but the contributor should be explicit.

- **Canonical example**: `outcome_falsification.py` `run()` loop catches `Exception` to keep going on individual formula failures. A targeted tuple `(ValueError, PatsyError, RuntimeError)` would still keep the loop going while letting unexpected `AttributeError`s surface.
- This is partly enforceable by Ruff `BLE001` — if the rule isn't enabled and the same pattern keeps appearing, file an issue to enable it with allowlist.

### SF-3: Docstring describes one quantity, code/test asserts another

Subtle bug-source. The docstring becomes the spec; if it's wrong, future contributors will write code matching the wrong spec.

- **Canonical example**: PR #826 `min_gap` docstring described "candidate-list distance" but the test asserted "actual time distance". These are different. Either the docstring or the test is wrong.
- **How to spot**: read the docstring of any new parameter, then read the test that exercises it. Confirm they describe the same quantity.

### SF-4: Default value is silently a no-op

A default that doesn't actually constrain anything is misleading. Either the default should impose a useful constraint, or the docstring should be honest about the no-op.

- **Canonical example**: `min_gap=1` with sampling-without-replacement is trivially true for any two distinct integer positions. Either change the default to `intervention_length` or document the no-op.
- **How to spot**: read default values; ask "what does this default actually do?" If the answer is "nothing", flag.

### SF-5: Algorithmic invariant not enforced where the rest of the code assumes it

When downstream code assumes invariant X (independence, non-overlap, sortedness) but upstream selection doesn't enforce X, X gets violated silently.

- **Canonical example**: `PlaceboInTime` random folds didn't enforce non-overlap of pseudo-windows; downstream hierarchical model assumes folds are exchangeable. Need either an `allow_overlap=False` default or an explicit docstring warning.
- **How to spot**: read the assumptions of downstream consumers, then read the upstream selector. Mismatch = flag.

### SF-6: Function ignores a parameter for a subset of input types

Polymorphic input handling that silently drops requested behaviour for one branch.

- **Canonical example**: `_draw_expected_effect_samples(n)` returned the user's numpy array verbatim, ignoring `n` entirely. Downstream code cycled with `i % len(prior)`, masking the silent drop.
- **How to spot**: any function that takes both a "size" parameter and a polymorphic input. Read each branch and confirm `n` is honoured (or explicitly documented as a no-op for that branch).

## Nits

Cosmetic / quality-of-life. Cluster these at the end of a review so they don't dilute attention from the must-fix items.

### N-1: `__repr__` shows defaults

`__repr__` should show non-default values only, mirroring `dataclass`-style brevity. If a sibling class in the same module already does this, inconsistency is the flag.

### N-2: Defensive code that doesn't actually defend

`except (KeyError, IndexError, np.linalg.LinAlgError)` where `LinAlgError` is never actually raised by the code under the try block. Harmless but adds noise.

- **How to spot**: for each exception in a multi-exception except clause, ask "what code path raises this?" If unclear, drop it.

### N-3: Test uses an unrealistic failure mode

Tests should exercise the failure modes users will actually hit, not synthetic ones. A test that uses a syntactically-invalid formula instead of a missing-column formula is testing the wrong thing.

### N-4: Docstring missing a non-obvious behaviour note

Behaviour that isn't bug-level but isn't intuitive — e.g. "this default halves the eligible window when X is unset." A one-line docstring note prevents future surprise.

### N-5: External data dependency in docs notebook

Notebook fetches from `https://...` at runtime. Brittle to URL changes / outages. Either bundle as a CausalPy example dataset or add a note.

### N-6: Test setup duplication ≥ 60 lines across ≥ 3 tests

A pytest fixture would consolidate. Not a bug, but an ongoing maintenance tax.

### N-7: Helper used N times in a notebook is library material

If a docs notebook defines a non-trivial helper (≥50 lines) and uses it ≥3 times, the helper is general, not example-specific. Suggest promotion to the library — usually as a follow-up PR rather than expanding the current one.

- **Canonical example**: `plot_placebo_calibration` in PR #826's notebook (~130 lines, called 3x). Inline review left the suggestion with a "happy for follow-up PR" disposition.
- See [code-patterns.md § Helper promotion](code-patterns.md#helper-promotion) for the cleanup checklist before promoting.

## Process patterns

Not code-level, but worth flagging on the PR.

### P-1: Mega-payload PR

Multiple logically-independent themes in one PR (interrogate config + new model `_clone()` + new check + plot bug fix). Slows review and complicates revert.

- **How to handle**: don't block on this for the current PR (sunk cost), but flag it once for future PRs.
- **Canonical example**: PR #826 issue-thread comment.

### P-2: Branch contains commits superseded by recently-merged work on `main`

Contributor's branch has commits that overlap with — and are slightly worse than — a fix that just landed on `main`. Need to identify the colliding commits, recommend rebase + drop, and confirm the canonical fix is picked up.

- **Canonical example**: PR #826 had two `panel_regression.py` commits superseded by #853.
- **How to spot**: when a PR is in `CONFLICTING` state, scan its file list against the most recent N commits to `main` for overlap.

### P-3: Notebook orphaned from `toctree`

A new notebook in `docs/source/notebooks/` not referenced anywhere in the docs tree. Sphinx renders it but there's no path for users to reach it.

- This is a hook candidate (under discussion); until the hook lands, look for it manually on every PR that adds a notebook.
