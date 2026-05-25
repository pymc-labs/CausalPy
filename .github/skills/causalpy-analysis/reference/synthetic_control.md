# Synthetic Control

Constructs a synthetic counterfactual from weighted control (donor) units.

```python
cp.SyntheticControl(
    data,                          # pd.DataFrame (panel data)
    treatment_time,                # int, float, or pd.Timestamp
    control_units,                 # list[str]: column names of donors
    treated_units,                 # list[str]: column names of treated unit(s)
    model=None,                    # typically WeightedSumFitter
    min_donor_correlation=0.0,     # minimum correlation filter for donors
    **kwargs
)
```

**How it works:**

1. Learns donor weights using pre-intervention data only
2. Applies weights to donors in post-intervention period → synthetic counterfactual
3. Impact: observed treated minus synthetic counterfactual

**Requirements:**

- Treated unit must lie within convex hull of donors
- Good pre-fit indicates valid synthetic control
- Use `min_donor_correlation` to filter weak donors

## Example

```python
import causalpy as cp

df = cp.load_data("sc")
result = cp.SyntheticControl(
    df,
    treatment_time=70,
    control_units=["a", "b", "c", "d", "e"],
    treated_units=["actual"],
    model=cp.pymc_models.WeightedSumFitter(
        sample_kwargs={"draws": 2000, "random_seed": 42}
    ),
)
result.plot()
result.print_coefficients()  # shows donor weights
result.effect_summary()
```

## OLS variant

```python
from causalpy.skl_models import WeightedProportion

result = cp.SyntheticControl(
    df,
    treatment_time=70,
    control_units=["a", "b", "c", "d", "e"],
    treated_units=["actual"],
    model=WeightedProportion(),
)
```

## Recommended checks

- `ConvexHullCheck` — verify treated unit is in donor convex hull
- `LeaveOneOut` — jackknife sensitivity across donors
- `PlaceboInSpace` — permutation test across units
