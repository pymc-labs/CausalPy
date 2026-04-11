---
name: example-datasets
description: Loads built-in CausalPy example datasets for demos, tutorials, and testing. Use when the user needs example data or asks about available demo datasets.
---

# Example Datasets

CausalPy ships with 20 built-in datasets for demos, tutorials, and testing.

## Usage

```python
import causalpy as cp
df = cp.load_data("dataset_name")
```

## Available Datasets

| Key | Method | Description |
|---|---|---|
| `"did"` | DiD | Generic difference-in-differences |
| `"banks"` | DiD | Banking closures |
| `"its"` | ITS | Generic interrupted time series |
| `"its simple"` | ITS | Simplified ITS |
| `"covid"` | ITS | Deaths & temperature (England/Wales) |
| `"sc"` | SC | Generic synthetic control |
| `"brexit"` | SC | UK GDP for Brexit causal impact |
| `"rd"` | RD | Generic regression discontinuity |
| `"drinking"` | RD | Minimum legal drinking age |
| `"geolift1"` | SC/DiD | Single treatment geo-lift |
| `"geolift_multi_cell"` | SC/DiD | Multi-cell geo-lift |
| `"anova1"` | PrePostNEGD | Pre/post ANCOVA nonequivalent groups |
| `"risk"` | IV | Acemoglu, Johnson & Robinson (2001) |
| `"schoolReturns"` | IV | Schooling returns |
| `"nhefs"` | IPW | National Health and Nutrition Survey |
| `"lalonde"` | IPW | LaLonde propensity score data |
| `"nets"` | IPW | National Supported Work Demonstration |
| `"pisa18"` | Various | PISA 2018 sample |
| `"nevo"` | Various | Nevo dataset |
| `"zipcodes"` | Geo | Geo-experimentation zipcode data |
