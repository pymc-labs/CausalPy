# Causal Demos

This skill handles the retrieval and loading of example datasets provided within the `CausalPy` library.

## Loading Data

To load internal datasets, use the `load_data` function from the `causalpy` module.

```python
import causalpy as cp

# Load a specific dataset
df = cp.load_data("dataset_name")
```

## Available Datasets

The following datasets are available for demonstration purposes. Use the key (left) to load the corresponding file.

| Key | Description / Context |
| :--- | :--- |
| `did` | Generic Difference-in-Differences example |
| `its` | Generic Interrupted Time Series example |
| `sc` | Generic Synthetic Control example |
| `banks` | DiD example (Banks) |
| `brexit` | Synthetic Control example (GDP impact of Brexit) |
| `covid` | ITS example (Deaths and temps in England/Wales) |
| `drinking` | Regression Discontinuity example (Minimum Legal Drinking Age) |
| `rd` | Generic Regression Discontinuity example |
| `its simple` | Simple ITS example |
| `anova1` | Generated ANCOVA example |
| `geolift1` | GeoLift example (Single cell) |
| `geolift_multi_cell` | GeoLift example (Multi cell) |
| `risk` | AJR2001 dataset |
| `nhefs` | NHEFS dataset |
| `schoolReturns` | Schooling Returns dataset |
| `pisa18` | PISA 2018 Sample Scale |
| `nets` | Nets DataFrame |
| `lalonde` | Lalonde dataset |

## Usage Example

```python
import causalpy as cp

# Load the 'did' dataset for a Difference-in-Differences demo
df = cp.load_data("did")
print(df.head())
```
