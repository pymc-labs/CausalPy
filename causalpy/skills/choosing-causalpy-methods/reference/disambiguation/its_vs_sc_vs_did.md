# ITS vs Synthetic Control vs DiD

Use this card when the request involves an intervention over time but the data topology is not yet clear.

## Deciding Question

What are the controls: no controls, comparison/control series used as predictors, donor time series used for synthetic weights, or treated/control groups observed over time?

## Choose `InterruptedTimeSeries`

- There is one aggregate or treated outcome series.
- There is a known intervention time.
- There are no treated/control group rows.
- The counterfactual comes from extrapolating the pre-intervention trend, optionally with comparison/control series as predictors.
- If comparison/control series are included as formula predictors, this is Comparative Interrupted Time Series (CITS) implemented with `InterruptedTimeSeries`.

## Choose `SyntheticControl`

- The data is a wide unit-by-time panel where columns are units.
- There is at least one treated unit and multiple untreated donor units.
- The counterfactual comes from a weighted donor combination that fits the treated unit in the pre-period.
- Use this instead of CITS when the user wants constrained donor weighting rather than regression-style comparison-series predictors.

## Choose `DifferenceInDifferences`

- The data has treated and control groups observed before and after the same intervention timing.
- The estimand is the treated-group by post-period contrast.
- The causal story rests on parallel trends, not donor-weight reconstruction.

## Choose Neither

- Use `StaggeredDifferenceInDifferences` when units adopt treatment at different times.
- Use `SyntheticDifferenceInDifferences` when the donor-panel design explicitly requires pre-period time weights as well as unit weights.
