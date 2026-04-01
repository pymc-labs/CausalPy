---
name: performing-causal-analysis
description: Fits causal models, estimates impacts, and plots results using CausalPy. Use when performing analysis with DiD, ITS, SC, or RD.
---

# Performing Causal Analysis

Executes causal analysis using CausalPy experiment classes.

## Workflow

1.  **Load Data**: Ensure data is in a Pandas DataFrame.
2.  **Initialize Experiment**: Use the appropriate class (see References).
3.  **Fit & Model**: Models are fitted automatically upon initialization if arguments are provided.
4.  **Analyze Results**: Use `summary()`, `print_coefficients()`, and `plot()`.

## Core Methods

*   `experiment.summary()`: Prints model summary and main results.
*   `experiment.plot()`: Visualizes observed vs. counterfactual.
*   `experiment.print_coefficients()`: Shows model coefficients.

## References

Detailed usage for specific methods:
*   [Difference-in-Differences](reference/diff_in_diff.md)
*   [Interrupted Time Series](reference/interrupted_time_series.md)
*   [Synthetic Control](reference/synthetic_control.md)
