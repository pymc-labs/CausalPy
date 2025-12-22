---
name: running-placebo-analysis
description: Performs placebo-in-time sensitivity analysis to validate causal claims. Use when checking model robustness, verifying lack of pre-intervention effects, or ensuring observed effects are not spurious.
---

# Running Placebo Analysis

Executes placebo-in-time sensitivity analysis to validate causal experiments.

## Workflow

1.  **Define Experiment Factory**: Create a function that returns a fitted CausalPy experiment (e.g., ITS, DiD, SC) given a dataset and time boundaries.
2.  **Configure Analysis**: Initialize `PlaceboAnalysis` with the factory, dataset, intervention dates, and number of folds (cuts).
3.  **Run Analysis**: Execute `.run()` to fit models on pre-intervention data folds.
4.  **Evaluate Results**: Compare placebo effects (which should be null) to the actual intervention effect. Use histograms and hierarchical models to quantify the "status quo" distribution.

## Key Concepts

*   **Placebo-in-time**: Simulating an intervention at a time when none occurred to check if the model falsely detects an effect.
*   **Fold**: A slice of pre-intervention data used to test a placebo period.
*   **Factory Pattern**: Decouples the placebo logic from the specific CausalPy experiment type.

## References

*   [Placebo-in-time Implementation](reference/placebo_in_time.md): Full code for the `PlaceboAnalysis` class, usage examples, and hierarchical status-quo modeling.
