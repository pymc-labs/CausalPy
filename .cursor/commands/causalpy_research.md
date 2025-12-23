# Causal Research

This skill is designed to help users "think deep" about their causal inference problem and select the most appropriate method.

## Decision Framework

When deciding on a method, ask the following questions:

1.  **Do you have a control group?**
    *   **Yes**: Proceed to check the structure of the control/treatment units.
    *   **No**: Consider methods that rely on time-series projection (e.g., Interrupted Time Series).

2.  **What is the unit structure?**
    *   **Single Treated Unit**:
        *   With multiple control units: **Synthetic Control (SCG)** is often best.
        *   With no control units: **Interrupted Time Series (ITS)**.
    *   **Multiple Treated Units**:
        *   With a control group: **Difference-in-Differences (DiD)**.

3.  **What is the time structure?**
    *   **Panel Data**: Data observed over time for multiple units. Required for DiD and SCG.
    *   **Time Series**: Data observed over time for a single unit (or aggregated units). Required for ITS.

## Method Selection Guide

### Difference-in-Differences (DiD)
*   **Best for**: Measuring the effect of a treatment by comparing the change in outcome over time between a treatment group and a control group.
*   **Key Assumption**: **Parallel Trends**. In the absence of treatment, the difference between the treatment and control group is constant over time.
*   **Data Requirement**: Panel data with a clear pre/post intervention period and a defined control group.

### Interrupted Time Series (ITS)
*   **Best for**: Evaluating the effect of an intervention on a single unit (or population) by analyzing the change in level and trend of the outcome after the intervention.
*   **Key Assumption**: The pre-intervention trend would have continued unchanged in the absence of the intervention.
*   **Data Requirement**: High-frequency time-series data (e.g., daily, monthly) with a clear intervention date.

### Synthetic Control (SCG)
*   **Best for**: Estimating the effect of an intervention on a single treated unit (e.g., a city, state, or country) using a weighted combination of control units.
*   **Key Assumption**: The control units can accurately reconstruct the treated unit's pre-intervention trajectory.
*   **Data Requirement**: Panel data with a long pre-intervention period and several potential control units that were not affected by the treatment.

## Diagnostic Questions to Ask the User

If the user is unsure, ask:
1.  "Does your data include a group that was *never* treated?"
2.  "Do you have data collected over time (e.g., days, months, years)?"
3.  "Is the treatment applied to a single entity (e.g., one store, one country) or many?"
4.  "Do you suspect other events happened at the same time as the treatment that could affect the outcome?"
