# Pipeline

Chain analysis steps into a reproducible workflow. Validates all steps before execution.

## Core Classes

```python
cp.Pipeline(data: pd.DataFrame, steps: list[Step])
```

**Methods:**
- `validate()` — validate all steps before execution
- `run()` → `PipelineResult` — execute steps sequentially

**Result:**
- `result.experiment` — fitted experiment
- `result.effect_summary` — EffectSummary object
- `result.sensitivity_results` — list of CheckResult
- `result.report` — HTML report string

## Available Steps

### EstimateEffect

Fits a causal experiment.

```python
cp.EstimateEffect(
    method=cp.InterruptedTimeSeries,    # experiment class (not instance)
    treatment_time=pd.Timestamp("2020-03-01"),
    formula="y ~ 1 + t",
    model=cp.pymc_models.LinearRegression(),
    # ... any other experiment kwargs
)
```

### SensitivityAnalysis

Runs one or more checks on the fitted experiment.

```python
cp.SensitivityAnalysis(
    checks=[
        cp.checks.PlaceboInTime(n_folds=4),
        cp.checks.OutcomeFalsification(outcome_column="y", falsification_column="z"),
    ]
)
```

### SensitivitySummary

Summarizes all sensitivity/diagnostic results.

```python
cp.SensitivitySummary(checks=None)  # None = summarize all
```

### GenerateReport

Generates a self-contained HTML report.

```python
cp.GenerateReport(
    include_plots=True,
    include_effect_summary=True,
    include_sensitivity=True,
    output_file="report.html",  # optional: save to file
)
```

## Full Example

```python
import pandas as pd
import causalpy as cp

df = cp.load_data("its")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

result = cp.Pipeline(
    data=df,
    steps=[
        cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            treatment_time=pd.Timestamp("2017-01-01"),
            formula="y ~ 1 + t",
            model=cp.pymc_models.LinearRegression(
                sample_kwargs={"draws": 1000, "random_seed": 42}
            ),
        ),
        cp.SensitivityAnalysis(
            checks=[
                cp.checks.PlaceboInTime(n_folds=4),
                cp.checks.PriorSensitivity(
                    prior_range={"sigma": [0.5, 1.0, 2.0]}
                ),
            ],
        ),
        cp.SensitivitySummary(),
        cp.GenerateReport(output_file="its_report.html"),
    ],
).run()

# Access results
print(result.effect_summary.text)
for sr in result.sensitivity_results:
    print(f"{sr.check_name}: {'PASSED' if sr.check_passed else 'FAILED'}")
```

## Pipeline Context

Internally, steps communicate via `PipelineContext`:
- `context.data` — the DataFrame
- `context.experiment` — fitted experiment (set by EstimateEffect)
- `context.experiment_config` — experiment configuration dict
- `context.effect_summary` — EffectSummary object
- `context.sensitivity_results` — accumulated check results
- `context.report` — generated report
