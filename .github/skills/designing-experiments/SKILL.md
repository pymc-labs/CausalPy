---
name: designing-experiments
description: Selects the appropriate quasi-experimental method (DiD, ITS, SC) based on data structure and research questions. Use when the user is unsure which method to apply.
---

# Designing Experiments

Helps select the appropriate causal inference method.

## Decision Framework

1.  **Control Group?**
    *   **Yes**: Go to Step 2.
    *   **No**: Consider **Interrupted Time Series (ITS)**.

2.  **Unit Structure?**
    *   **Single Treated Unit**:
        *   With multiple controls: **Synthetic Control (SC)**.
        *   No controls: **ITS**.
    *   **Multiple Treated Units**:
        *   With control group: **Difference-in-Differences (DiD)**.

3.  **Time Structure?**
    *   **Panel Data** (Multiple units over time): Required for DiD and SC.
    *   **Time Series** (Single unit over time): Required for ITS.

## Method Quick Reference

*   **Difference-in-Differences (DiD)**: Compares trend changes between treated and control groups. Assumes **Parallel Trends**.
*   **Interrupted Time Series (ITS)**: Analyzes trend/level change for a single unit after intervention. Assumes **Trend Continuity**.
*   **Synthetic Control (SC)**: Constructs a synthetic counterfactual from weighted control units. Assumes **Convex Hull** (treated unit within range of controls).
