# Method Selection

## Decision Framework

### 1. What data structure do you have?

| Structure | Methods |
|---|---|
| Panel data (units x time, with control group) | DiD, Staggered DiD, Synthetic Control |
| Single time series (one unit, before/after) | ITS, PiecewiseITS |
| Cross-sectional with running variable | RD, RK |
| Cross-sectional with treatment selection | IPW, IV |
| Pre/post with non-equivalent groups | PrePostNEGD |

### 2. Do you have a control group?

- **Yes, multiple units**: DiD or Staggered DiD
- **Yes, donors for synthetic counterfactual**: Synthetic Control
- **No, single unit over time**: ITS or PiecewiseITS
- **No, but have a running variable**: RD or RK
- **No, but have instruments**: IV

### 3. Treatment timing?

- **Single, sharp treatment time**: DiD, ITS, SC
- **Staggered adoption across units**: Staggered DiD
- **Multiple intervention points**: PiecewiseITS (with `step()` and `ramp()` transforms)
- **Threshold-based assignment**: RD or RK

## All 9 Methods

| Method | Class | OLS | Bayes | Key Assumption |
|---|---|---|---|---|
| Difference-in-Differences | `DifferenceInDifferences` | Yes | Yes | Parallel trends |
| Staggered DiD | `StaggeredDifferenceInDifferences` | Yes | Yes | Parallel trends + absorbing treatment |
| Interrupted Time Series | `InterruptedTimeSeries` | Yes | Yes | Trend continuity |
| Piecewise ITS | `PiecewiseITS` | Yes | Yes | Trend continuity between breaks |
| Synthetic Control | `SyntheticControl` | Yes | Yes | Convex hull, donor relevance |
| Regression Discontinuity | `RegressionDiscontinuity` | Yes | Yes | Continuity at threshold |
| Regression Kink | `RegressionKink` | No | Yes | Slope continuity at kink |
| Inverse Propensity Weighting | `InversePropensityWeighting` | No | Yes | No unmeasured confounders |
| Instrumental Variable | `InstrumentalVariable` | No | Yes | Exclusion restriction, relevance |
| PrePost NEGD | `PrePostNEGD` | Yes | Yes | Selection on observables |

## Matching Methods to Checks

| Method | Recommended Checks |
|---|---|
| DiD | `PreTreatmentPlaceboCheck`, `PlaceboInTime`, `OutcomeFalsification` |
| ITS | `PlaceboInTime`, `PriorSensitivity`, `PersistenceCheck` |
| SC | `PlaceboInSpace`, `LeaveOneOut`, `ConvexHullCheck` |
| RD | `BandwidthSensitivity`, `McCraryDensityTest` |
| IPW | `PriorSensitivity`, `OutcomeFalsification` |
| IV | `PriorSensitivity` |
