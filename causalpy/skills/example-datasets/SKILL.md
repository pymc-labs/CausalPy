---
name: example-datasets
description: Load built-in CausalPy example datasets for demos, tutorials, tests, and quick causal-analysis prototypes. Use when the user needs sample data or asks which demo datasets are available.
---

# Example Datasets

CausalPy ships with built-in datasets that can be loaded with `cp.load_data(...)`.

## Usage

```python
import causalpy as cp

df = cp.load_data("did")
```

## Available Datasets

| Key | Typical use | Description |
|---|---|---|
| `"did"` | Difference-in-differences | Synthetic DiD example data |
| `"banks"` | Difference-in-differences | Historic banking closures data |
| `"its"` | Interrupted time series | Seasonal synthetic ITS data |
| `"its simple"` | Interrupted time series | Simplified synthetic ITS data |
| `"covid"` | Interrupted time series | Deaths and temperature data for England and Wales |
| `"sc"` | Synthetic control | Synthetic control example data |
| `"brexit"` | Synthetic control | UK GDP data for Brexit causal impact |
| `"california_prop99"` | Synthetic control | California Proposition 99 cigarette sales panel |
| `"rd"` | Regression discontinuity | Synthetic RD example data |
| `"drinking"` | Regression discontinuity | Minimum legal drinking age data |
| `"geolift1"` | Geo experiments | Single-treatment geo-lift data |
| `"geolift_multi_cell"` | Geo experiments | Multi-cell geo-lift data |
| `"anova1"` | PrePostNEGD | Pre/post nonequivalent groups example |
| `"risk"` | Instrumental variables | Acemoglu, Johnson, and Robinson institutions data |
| `"schoolReturns"` | Instrumental variables | Schooling returns data |
| `"nhefs"` | Inverse propensity weighting | National Health and Nutrition Examination Survey data |
| `"lalonde"` | Inverse propensity weighting | LaLonde propensity-score data |
| `"nets"` | Inverse propensity weighting | National Supported Work Demonstration data |
| `"pisa18"` | General examples | PISA 2018 sample data |
| `"nevo"` | General examples | Berry, Levinsohn, and Pakes cereal data |
| `"zipcodes"` | Geo experiments | Zipcode-level geo-experiment data |

## Guidance

- Prefer these bundled datasets for examples and docs instead of fetching data at runtime.
- For method selection, use `choosing-causalpy-methods` after identifying the data shape.
- For fitting and plotting, use `running-causalpy-experiments`.
