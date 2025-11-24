## Summary

Extend the `InterruptedTimeSeries` class to support a three-period design: **pre-intervention**, **intervention**, and **post-intervention** periods. This enables analysis of temporary interventions and measurement of long-term effect persistence after interventions end.

## Motivation

### Current Limitation

The current ITS implementation assumes interventions are permanent—once the intervention starts, it continues indefinitely. This works well for permanent policy changes or structural interventions but is limiting for:

- Temporary marketing campaigns
- Time-limited policy trials
- Seasonal programs
- Clinical trials with fixed treatment duration
- Lockdowns or temporary restrictions

### Why This Matters

Many real-world interventions are **temporary**, and decision-makers need to answer questions like:

1. **"What was the immediate effect during the intervention?"** - Measures direct causal impact while intervention was active
2. **"Did the effect persist after the intervention ended?"** - Quantifies lasting behavioral or structural changes
3. **"How much of the effect decayed vs. remained?"** - Informs ROI and cost-benefit analysis
4. **"How long did effects last?"** - Guides optimal intervention duration

### Use Cases

- **Marketing**: Ad campaign with lasting brand awareness
- **Medicine**: Treatment effects that persist after medication stops
- **Education**: Training programs with skill retention
- **Environmental**: Temporary pollution controls with ecosystem recovery
- **Public policy**: Time-limited trials with behavioral habit formation

## How It Works (High-Level Logic)

The core statistical approach **remains unchanged**—we still fit the model only on pre-intervention data. The extension is in how we analyze the post-treatment period.

### Current Two-Period Design

```
Timeline:
|-------- Pre --------|---------- Post (Intervention) ----------|
         (fit)                    (forecast)
```

1. Fit model on pre-intervention period
2. Forecast counterfactual for everything after intervention starts
3. Calculate causal impact: `actual - counterfactual`

### Proposed Three-Period Design

```
Timeline:
|-------- Pre --------|------ Intervention ------|------- Post --------|
         (fit)           (forecast period 1)      (forecast period 2)
```

1. Fit model on pre-intervention period (unchanged)
2. Forecast counterfactual for everything after intervention starts (unchanged)
3. **NEW**: Split the forecast and impact into two labeled periods:
   - **Intervention period**: `treatment_time` to `treatment_end_time`
   - **Post-intervention period**: `treatment_end_time` onward

4. Analyze impacts separately:
   - `intervention_impact = actual - counterfactual` (during intervention)
   - `post_intervention_impact = actual - counterfactual` (after intervention)
   - `persistence_ratio = mean(post_intervention_impact) / mean(intervention_impact)`

### Key Insight

We make **one continuous counterfactual forecast** using the pre-intervention model, then **slice it into two periods** for analysis. No additional statistical complexity—just more informative decomposition of effects.

## Implementation Approach

### API Design (Backward Compatible)

```python
# Current usage (still works exactly the same)
result = cp.InterruptedTimeSeries(
    data,
    treatment_time="2024-01-01",
    formula="y ~ 1 + t + C(month)",
    model=cp.pymc_models.LinearRegression()
)

# New usage (opt-in via treatment_end_time parameter)
result = cp.InterruptedTimeSeries(
    data,
    treatment_time="2024-01-01",
    treatment_end_time="2024-04-01",  # NEW: optional parameter
    formula="y ~ 1 + t + C(month)",
    model=cp.pymc_models.LinearRegression()
)
```

### New Attributes

When `treatment_end_time` is provided, additional attributes become available:

```python
# Existing attributes (always available)
result.post_pred     # Counterfactual forecast from treatment_time onward
result.post_impact   # Causal impact from treatment_time onward

# New attributes (only when treatment_end_time is provided)
result.data_intervention              # Data during intervention period
result.data_post_intervention         # Data after intervention ends
result.intervention_pred              # Counterfactual during intervention
result.post_intervention_pred         # Counterfactual post-intervention
result.intervention_impact            # Impact during intervention
result.post_intervention_impact       # Impact after intervention ends
result.intervention_impact_cumulative # Cumulative during intervention
result.post_intervention_impact_cumulative # Cumulative post-intervention
```

### New Methods

```python
# Analyze effect persistence
result.analyze_persistence()
# Returns:
# {
#   'mean_effect_during': 50.0,
#   'mean_effect_post': 15.0,
#   'persistence_ratio': 0.30,  # 30% of effect remained
#   'total_effect_during': 600.0,
#   'total_effect_post': 210.0
# }

# Test if post-intervention effect is significantly different from zero
result.test_permanent_effect()  # Returns posterior probability for Bayesian models

# Fit parametric decay model to post-intervention impacts
result.fit_decay_model(decay_type='exponential')  # or 'linear', 'step'
```

### Enhanced Effect Summary Reporting

The existing `effect_summary()` method provides decision-ready reports with average/cumulative effects, HDIs, tail probabilities, and relative effects. For the three-period design, this will be extended to provide separate summaries for each period:

```python
# Current usage (two-period): summary for entire post-period
stats = result.effect_summary()
print(stats.text)
# "Post-period (...), the average effect was 1.83 (95% HDI [0.66, 2.94])..."

# New usage (three-period): get summaries for specific periods
stats_intervention = result.effect_summary(period='intervention')
stats_post = result.effect_summary(period='post')

print(stats_intervention.text)
# "During intervention (2024-01-01 to 2024-04-01), the average effect was
#  50.2 (95% HDI [45.1, 55.3]), with a posterior probability of an increase
#  of 0.999. The cumulative effect was 603.2 (95% HDI [541.2, 663.6])..."

print(stats_post.text)
# "Post-intervention (2024-04-01 to 2024-12-31), the average effect was
#  15.3 (95% HDI [10.2, 20.4]), with a posterior probability of an increase
#  of 0.985. The cumulative effect was 214.2 (95% HDI [142.8, 285.6])..."

# Comparative summary showing persistence
stats_comparison = result.effect_summary(period='comparison')
print(stats_comparison.text)
# "Effect persistence: The post-intervention effect (15.3, 95% HDI [10.2, 20.4])
#  was 30.5% of the intervention effect (50.2, 95% HDI [45.1, 55.3]), with a
#  posterior probability of 0.95 that some effect persisted beyond the
#  intervention period."
```

**API Extension:**
- `period='intervention'`: Summary for intervention period only
- `period='post'`: Summary for post-intervention period only
- `period='comparison'`: Comparative summary with persistence metrics
- `period=None` (default): Maintains backward compatibility (summarizes all post-treatment data)

### Updated Plotting

Enhanced plots showing three periods with distinct visual treatment:

```python
fig, ax = result.plot()
# Creates similar 3-panel plot:
# 1. Observed vs counterfactual (all three periods)
# 2. Instantaneous impact during intervention. Instantaneous impact post-intervention (shows decay)
# 3. Cumulative impact (both periods)
```

### Enhanced Summary Output

```python
result.summary()

# Output:
# ========================Interrupted Time Series========================
# Formula: y ~ 1 + t + C(month)
# Pre-intervention: 2023-01-01 to 2023-12-31
# Intervention: 2024-01-01 to 2024-04-01
# Post-intervention: 2024-04-01 to 2024-12-31
#
# Impact during intervention: 50.2 (94% HDI [45, 55])
# Impact post-intervention: 15.3 (94% HDI [10, 20])
# Persistence ratio: 30.5%
#
# Model coefficients:
# ...
```

## Implementation Details

### Core Changes Required

1. **Signature update** in `interrupted_time_series.py`:
   ```python
   def __init__(
       self,
       data: pd.DataFrame,
       treatment_time: Union[int, float, pd.Timestamp],
       formula: str,
       model: Union[PyMCModel, RegressorMixin] | None = None,
       treatment_end_time: Union[int, float, pd.Timestamp] | None = None,  # NEW
       **kwargs: dict,
   )
   ```

2. **Add `_split_post_period()` method** to slice forecasts/impacts:
   ```python
   def _split_post_period(self):
       """Split post period into intervention and post-intervention."""
       if self.treatment_end_time is None:
           return

       # Create masks based on indices
       during_mask = self.datapost.index < self.treatment_end_time
       post_mask = self.datapost.index >= self.treatment_end_time

       # Slice predictions, impacts, and data using masks
       # (Works for both PyMC xarray and sklearn numpy arrays)
   ```

3. **Update plotting methods** to detect and visualize three periods

4. **Add persistence analysis methods**

5. **Update input validation** to ensure `treatment_end_time > treatment_time`

### What Stays the Same

- ✅ All model fitting logic (fit on pre-intervention only)
- ✅ All prediction/forecast logic (one continuous forecast)
- ✅ All existing attributes remain unchanged
- ✅ All existing tests pass without modification
- ✅ Perfect backward compatibility

## Testing Requirements

1. Test three-period workflow with PyMC models
2. Test three-period workflow with sklearn models
3. Test backward compatibility (`treatment_end_time=None`)
4. Test persistence calculations
5. Test edge cases (very short post-period, treatment_end_time at data boundary)
6. Test with datetime and numeric indices

## Documentation Requirements

1. **New example notebook**: `its_three_period_pymc.ipynb`
   - Simulate data with temporary intervention and decay dynamics
   - Demonstrate persistence analysis
   - Real-world example (e.g., marketing campaign, policy trial)

2. **Update existing notebooks**: Add note about `treatment_end_time` parameter

3. **API documentation**: Document new parameters, attributes, and methods

4. **Glossary additions**: Define "effect persistence", "effect decay", "intervention period"

## Example: Marketing Campaign

```python
import causalpy as cp

# Sales data with 3-month ad campaign
result = cp.InterruptedTimeSeries(
    data=sales_data,
    treatment_time="2024-01-01",     # Campaign starts
    treatment_end_time="2024-03-31",  # Campaign ends
    formula="sales ~ 1 + t + day_of_week",
    model=cp.pymc_models.LinearRegression()
)

# Visualize three periods
fig, ax = result.plot()

# Analyze persistence
persistence = result.analyze_persistence()
print(f"Campaign lift: ${persistence['mean_effect_during']:.0f}/week")
print(f"Lasting lift: ${persistence['mean_effect_post']:.0f}/week")
print(f"Persistence: {persistence['persistence_ratio']:.1%}")

# Output:
# Campaign lift: $50/week
# Lasting lift: $15/week
# Persistence: 30.0%
```

---

**Benefits**: Significantly expands the types of causal questions CausalPy can answer, particularly for temporary interventions which are extremely common in practice (marketing, clinical trials, policy experiments, etc.). Implementation is low-risk due to backward compatibility and minimal changes to core logic.
