---
name: loading-datasets
description: Loads internal CausalPy example datasets. Use when the user needs example data or asks about available demos.
---

# Loading Datasets

Loads example datasets provided with CausalPy.

## Usage

```python
import causalpy as cp
df = cp.load_data("dataset_name")
```

## Available Datasets

### Synthetic (generated programmatically)

| Key | Description |
| :--- | :--- |
| `did` | Difference-in-Differences example |
| `its` | Interrupted Time Series (seasonal) |
| `its simple` | Interrupted Time Series (simple trend) |
| `rd` | Regression Discontinuity example |
| `sc` | Synthetic Control example |
| `anova1` | ANCOVA / Pre-Post NEGD example |
| `geolift1` | GeoLift single treatment cell |
| `geolift_multi_cell` | GeoLift multi-cell experiment |

### Real-world (CSV)

| Key | Description |
| :--- | :--- |
| `banks` | Historic banking closures (DiD) |
| `brexit` | UK GDP in billions (Synthetic Control) |
| `covid` | Deaths and temperatures, England & Wales (ITS) |
| `drinking` | Minimum legal drinking age (Regression Discontinuity) |
| `risk` | Acemoglu, Johnson & Robinson 2001 (Instrumental Variables) |
| `nhefs` | National Health and Nutrition Examination Survey (IPW) |
| `schoolReturns` | Schooling returns (Instrumental Variables) |
| `lalonde` | LaLonde dataset (Propensity Score / IPW) |
| `nets` | National Supported Work Demonstration |
| `pisa18` | PISA 2018 sample data |
| `zipcodes` | Geo-experimentation zipcode data (Comparative ITS) |
| `nevo` | Berry, Levinsohn & Pakes 1995 cereal data |
| `california_prop99` | California Proposition 99 cigarette sales (Synthetic Control) |
