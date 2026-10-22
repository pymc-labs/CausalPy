# Recurring Review Patterns

Use this severity-sorted checklist during the diff pass. These patterns are grounded in CausalPy review experience and should stay focused on issues that need judgement rather than mechanical linting.

## Must-Fix Patterns

### MF-1: Subclass Override Drops Base-Class Behavior

A subclass overrides a method but does not forward behavior or kwargs from the base class.

- Canonical example: a PyMC model `_clone()` override that omits `priors=self._user_priors`, silently dropping user priors.
- How to spot: whenever a diff adds an override, read the base method side by side and compare kwargs, returned types, and side effects.

### MF-2: Constrained String Parameter Not Typed as `Literal`

Constrained string parameters should use `Literal`.

- How to spot: a new parameter is typed as `str`, but the docstring or implementation accepts a fixed set such as `"sequential"` or `"random"`.
- Ask for `Literal["sequential", "random"]` or the equivalent exact set.

### MF-3: Behavioral Change Without a Pinning Test

A bug fix or logic change needs a test that would fail under the old behavior and pass under the new behavior.

- How to spot: non-test code changes a formula, branch condition, array shape, default, indexing rule, or exception path without a corresponding assertion.
- Ask for a regression test tied to the changed behavior.

### MF-4: New Public API Inconsistent With Sibling Implementations

New public classes or functions should match nearby contracts unless the PR intentionally changes the design.

- How to spot: new experiment/model/check classes missing support flags, clone behavior, backend dispatch, return shapes, explicit plotting signatures, or project exceptions.
- Ask for consistency or a documented rationale.

## Should-Fix Patterns

### SF-1: Memory-Heavy Retainer Defaults On

Result/check classes that retain full fitted models, `InferenceData`, or large arrays by default impose hidden costs on common users.

- How to spot: result objects storing fitted experiments, posteriors, or generated figures by default.
- Ask whether summary-only should be default with opt-in storage.

### SF-2: Broad Exception Handling in Library Code

`except Exception` and bare `except:` hide unexpected bugs as well as expected recoverable failures.

- How to spot: broad catches in library code, especially loops over formulas, models, or datasets.
- Ask for a targeted exception tuple or an explicit rationale.

### SF-3: Docstring, Code, and Tests Describe Different Quantities

The docstring becomes the public spec. If tests assert a different definition, future changes will follow the wrong contract.

- How to spot: read every new parameter's docstring and the test that exercises it; confirm they describe the same quantity and units.

### SF-4: Default Value Is Silently a No-Op

A default that appears meaningful but imposes no constraint misleads users.

- Canonical example: a placebo-in-time selector with `min_gap=1` while sampling distinct integer positions may not constrain anything in practice; either use a meaningful default such as the intervention length or document that the default is only a placeholder.
- How to spot: ask "what does this default actually do?" for every new numeric or policy default.
- Ask for a meaningful default or honest documentation that it is a no-op.

### SF-5: Algorithmic Invariant Not Enforced

Downstream code often assumes invariants such as non-overlap, sortedness, independence, monotonicity, or aligned indexes. Upstream selection must enforce or document them.

- How to spot: read downstream assumptions, then read upstream selectors or validators. Mismatches need changes.

### SF-6: Parameter Ignored for One Input Type

Polymorphic input handling can silently drop requested behavior for one branch.

- How to spot: any function with both a size/shape parameter and polymorphic input. Confirm each branch honors the parameter or documents why it does not.

## Nits

### N-1: `__repr__` Shows Defaults

Prefer `__repr__` output that highlights non-default values only, especially when sibling classes follow that style.

### N-2: Defensive Code That Does Not Defend

Multi-exception catches should only name exceptions that the protected code can actually raise.

### N-3: Test Uses an Unrealistic Failure Mode

Tests should exercise realistic user failures, such as missing columns, bad indexes, or shape mismatches, rather than unrelated synthetic errors.

### N-4: Non-Obvious Behavior Missing a Docstring Note

If a behavior is surprising but intentional, add a short docstring note.

### N-5: External Data Dependency in a Docs Notebook

Runtime URL fetches in notebooks are brittle. Prefer bundled data or document and cite the external dependency.

### N-6: Repeated Test Setup Should Be a Fixture

Repeated setup across several tests is a maintenance smell. Suggest a fixture when it improves clarity without hiding important differences.

### N-7: Notebook Helper Looks Like Library Material

If a notebook defines a non-trivial helper and uses it repeatedly, suggest promotion to the library, usually as follow-up work.

## Process Patterns

### P-1: Mega-Payload PR

Multiple independent themes in one PR slow review and complicate revert. Usually flag for future process rather than blocking late-stage work.

### P-2: Branch Contains Work Superseded by Main

When a PR conflicts or overlaps with recent main commits, identify the colliding commits and recommend a rebase or commit drop.

### P-3: Notebook Orphaned From the Toctree

A new notebook not referenced from the docs tree is effectively hidden from users. Check this until a hook covers it.
