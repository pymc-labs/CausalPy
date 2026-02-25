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

| Key | Description |
| :--- | :--- |
| `did` | Generic Difference-in-Differences |
| `its` | Generic Interrupted Time Series |
| `sc` | Generic Synthetic Control |
| `banks` | DiD (Banks) |
| `brexit` | Synthetic Control (Brexit) |
| `covid` | ITS (Covid) |
| `drinking` | Regression Discontinuity (Drinking Age) |
| `rd` | Generic Regression Discontinuity |
| `geolift1` | GeoLift (Single cell) |
| `geolift_multi_cell` | GeoLift (Multi cell) |
