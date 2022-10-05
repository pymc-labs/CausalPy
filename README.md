# [this repo name is to be decided!]

This package focusses on Bayesian causal inference

## Examples

* [Synthetic control](notebooks/synthetic_control_generative.ipynb)
* Interrupted time series

## Synthetic control
This is appropriate when you have multiple units, one of which is treated. You build a synthetic control as a weighted combination of the untreated units.

| time | Treatment | Control 1 | Control 2 | Control 3 |
|------|-----------|-----------|-----------|-----------|
| 0    | $y_0$ | $x_{1,0}$ | $x_{2,0}$ | $x_{3,0}$ |
| 1    | $y_1$ | $x_{1,1}$ | $x_{2,1}$ | $x_{3,1}$ |
|$\ldots$ | $\ldots$  | $\ldots$  | $\ldots$  | $\ldots$  |
| N    | $y_N$ | $x_{1,N}$ | $x_{2,N}$ | $x_{3,N}$ |

![](img/synthetic_control.png)

## Interrupted time series
This is appropriate when you have a single treated unit, and therefore a single time series, and do _not_ have a set of untreated units.

| time | Treatment |
|------|-----------|
| 0    | $y_0$ |
| 1    | $y_1$ |
|$\ldots$ | $\ldots$  |
| N    | $y_N$ |

![](img/interrupted_time_series.png)

--- 

## Local development

1. Create conda environment:

```bash
conda create --name causal_impact_env --file requirements.txt
```

2. Activate environment:

```bash
conda activate causal_impact_env
```

3. Import the package

```bash
pip install -e ./
```