# ITS vs Piecewise ITS

Use this card when the user has one outcome time series and one or more known intervention times.

## Deciding Question

Is the counterfactual meant to be forecast from the pre-intervention period, or should the model estimate explicit level and slope changes in the full observed series?

## Choose `InterruptedTimeSeries`

- The data is one outcome series with a known intervention time and no donor pool.
- The estimand is a post-intervention gap between observed outcomes and a counterfactual forecast fit on pre-intervention data.
- The main risk is whether the pre-period trend, seasonality, and covariates would have remained valid without treatment.

## Choose `PiecewiseITS`

- The intervention dates are known and should appear directly in the formula through `step()` or `ramp()` terms.
- The estimand is an explicit level change, slope change, or cumulative effect from a segmented regression fit to the full series.
- Multiple known interruptions are part of the design.

## Choose Neither

- Use `SyntheticControl` if the user has a treated series plus donor units.
- Return Not implemented in CausalPy if the user needs unknown changepoint discovery rather than known interruptions.
