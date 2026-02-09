# Causal Demos

This skill handles the retrieval and loading of example datasets provided within the `CausalPy` library.

## Loading Data

To load internal datasets, use the `load_data` function from the `causalpy` module.

```python
import causalpy as cp

# Load a specific dataset
df = cp.load_data("dataset_name")
```

## Discovering Available Datasets

To discover available datasets and their descriptions, read the `load_data` docstring in `causalpy/data/datasets.py`:

```python
import causalpy as cp

# View the docstring with dataset descriptions
help(cp.load_data)
```

The docstring contains a complete list of dataset keys and descriptions of each dataset's intended use case.

## Usage Example

```python
import causalpy as cp

# Load the 'did' dataset for a Difference-in-Differences demo
df = cp.load_data("did")
print(df.head())
```
