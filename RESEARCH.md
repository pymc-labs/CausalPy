# RESEARCH.md: CausalPy Developer Guide

**Version:** 0.6.0
**Last Updated:** November 2025
**Purpose:** Internal architecture reference for contributors and developers

---

## 1. Purpose and Scope

### What is CausalPy?

CausalPy is a Python package focused on **causal inference in quasi-experimental settings**. It provides a unified interface for analyzing observational data using established causal inference methods, with support for both Bayesian (PyMC) and frequentist (scikit-learn) modeling approaches.

### Supported Causal Inference Methods

CausalPy currently supports 8 quasi-experimental designs:

1. **Difference in Differences (DiD)** - Compare changes over time between treatment and control groups
2. **Interrupted Time Series (ITS)** - Analyze time series before and after an intervention
3. **Regression Discontinuity (RD)** - Exploit sharp cutoffs in treatment assignment
4. **Regression Kink (RKink)** - Detect slope changes at a kink point rather than level jumps
5. **Synthetic Control (SC)** - Construct synthetic counterfactuals from control units
6. **Instrumental Variable (IV)** - Address endogeneity using instrumental variables
7. **Inverse Propensity Weighting (IPW)** - Weight observations by propensity scores
8. **Pre-Post NEGD** - Pre-post analysis with non-equivalent group designs (including ANCOVA)

### Dual Backend Philosophy

CausalPy is designed to work with **two statistical backends**:

- **PyMC (Bayesian)**: Full posterior distributions via MCMC sampling, HDI intervals, probabilistic statements
- **Scikit-learn (Frequentist/OLS)**: Point estimates with confidence intervals, faster computation

Users can switch between backends by providing different model objects to experiment classes. The same experiment class works with both backends.

### Explicit Non-Goals

- **Not a general causal inference library**: Focus is specifically on quasi-experimental designs, not general DAG-based causal inference
- **Not for randomized experiments**: Designed for observational data with quasi-experimental structure
- **Not for causal discovery**: Assumes the causal structure is known from domain knowledge
- **Not for structural causal models**: Focused on reduced-form approaches, not full structural estimation

---

## 2. High-Level Architecture

### Package Structure

```
causalpy/
├── __init__.py                    # Public API exports
├── experiments/                   # Experiment design implementations
│   ├── __init__.py
│   ├── base.py                    # BaseExperiment abstract class
│   ├── diff_in_diff.py            # Difference in Differences
│   ├── interrupted_time_series.py # Interrupted Time Series
│   ├── regression_discontinuity.py # Regression Discontinuity
│   ├── regression_kink.py         # Regression Kink
│   ├── synthetic_control.py       # Synthetic Control
│   ├── instrumental_variable.py   # Instrumental Variable
│   ├── inverse_propensity_weighting.py # IPW
│   └── prepostnegd.py             # Pre-Post NEGD/ANCOVA
├── pymc_models.py                 # PyMC model implementations
├── skl_models.py                  # Scikit-learn model adaptors
├── plot_utils.py                  # Visualization helper functions
├── reporting.py                   # Statistical summaries and prose generation
├── utils.py                       # General utility functions
├── custom_exceptions.py           # Domain-specific exceptions
├── data/                          # Example datasets
│   ├── __init__.py
│   ├── datasets.py                # Data loading functions
│   └── *.csv                      # Example CSV files
└── tests/                         # Test suite
    ├── conftest.py                # Pytest fixtures
    ├── test_*.py                  # Test modules
    └── ...
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `experiments/` | Implement causal inference designs; orchestrate model fitting, plotting, and reporting |
| `pymc_models.py` | Provide PyMC model classes with sklearn-like interface (`fit`, `predict`, `score`) |
| `skl_models.py` | Adapt sklearn regressors to work with CausalPy (add `calculate_impact()`, etc.) |
| `plot_utils.py` | Reusable plotting functions (HDI plots, counterfactual visualizations) |
| `reporting.py` | Generate statistical summaries and prose reports via `effect_summary()` |
| `utils.py` | General utilities (rounding, formula parsing, type checking) |
| `custom_exceptions.py` | Domain-specific exceptions (`FormulaException`, `DataException`, `BadIndexException`) |
| `data/` | Example datasets and loading utilities |

### Module Interactions

```
User Code
    ↓
Experiment Classes (experiments/)
    ↓
    ├── Models (pymc_models.py or skl_models.py)
    │       ↓
    │   Utils (utils.py)
    │
    ├── Plotting (plot_utils.py)
    │       ↓
    │   matplotlib/seaborn/arviz
    │
    └── Reporting (reporting.py)
            ↓
        Statistical summaries
```

**Typical Flow:**
1. User instantiates an experiment class with data, formula, and model
2. Experiment's `fit()` method:
   - Parses formula with patsy
   - Validates data (custom exceptions)
   - Calls model's `fit()` method
3. User calls `plot()` → dispatches to `_bayesian_plot()` or `_ols_plot()`
4. User calls `effect_summary()` → uses `reporting.py` to compute statistics

### Cross-Cutting Concerns

- **Data validation**: Custom exceptions (`FormulaException`, `DataException`, `BadIndexException`) provide clear error messages
- **Formula parsing**: Uses `patsy` library for R-style formulas (e.g., `"y ~ 1 + x1 + x2*x3"`)
- **Plotting style**: Applies `arviz-darkgrid` style context during plotting, then reverts
- **Data index naming**: Experiments expect data index to be named `"obs_ind"` for consistency
- **Type dispatching**: Many methods check `isinstance(model, PyMCModel)` vs `isinstance(model, RegressorMixin)` to dispatch to appropriate implementation

---

## 3. Core Abstractions and Class Structure

### BaseExperiment (`causalpy/experiments/base.py`)

The foundational abstract class for all quasi-experimental designs.

**Key Class Attributes:**
- `supports_bayes: bool` - Whether this experiment works with PyMC models
- `supports_ols: bool` - Whether this experiment works with sklearn models
- `labels: list[str]` - Coefficient labels for printing

**Constructor:**
```python
def __init__(self, model: Union[PyMCModel, RegressorMixin] | None = None) -> None
```
- Validates model compatibility with experiment's supported backends
- Wraps sklearn models with `create_causalpy_compatible_class()` if needed

**Abstract Methods** (must be implemented by subclasses):
- `fit(*args, **kwargs)` - Fit the model to data
- `_bayesian_plot(*args, **kwargs) -> tuple` - Plotting for PyMC models (only if `supports_bayes=True`)
- `_ols_plot(*args, **kwargs) -> tuple` - Plotting for sklearn models (only if `supports_ols=True`)
- `get_plot_data_bayesian(*args, **kwargs) -> pd.DataFrame` - Extract plot data for PyMC
- `get_plot_data_ols(*args, **kwargs) -> pd.DataFrame` - Extract plot data for sklearn

**Concrete Methods** (implemented in base class):
- `plot(*args, **kwargs) -> tuple` - Dispatches to `_bayesian_plot()` or `_ols_plot()` based on model type
- `print_coefficients(round_to: int | None = None)` - Pretty-print model coefficients
- `effect_summary(window, direction, alpha, cumulative, relative, min_effect, treated_unit) -> EffectSummary` - Generate comprehensive causal effect summary with statistics and prose
- `get_plot_data(*args, **kwargs) -> pd.DataFrame` - Dispatches to backend-specific method
- `idata` property - Returns `InferenceData` object (PyMC only)

**Design Pattern:**
BaseExperiment uses the **Template Method** pattern: `plot()` and `get_plot_data()` orchestrate the workflow and dispatch to backend-specific implementations.

### PyMCModel (`causalpy/pymc_models.py`)

Wrapper around `pm.Model` providing an sklearn-like interface.

**Inheritance:**
```python
class PyMCModel(pm.Model):
    ...
```

**Key Class Attributes:**
- `default_priors: Dict[str, Prior]` - Default prior specifications (can be overridden)

**Core Methods:**
- `priors_from_data(X, y) -> Dict[str, Prior]` - Generate data-adaptive priors (override in subclasses)
- `build_model(X, y, coords)` - Define PyMC model structure (abstract, must implement in subclasses)
- `fit(X, y, coords, **sample_kwargs) -> az.InferenceData` - Fit model via MCMC sampling
- `predict(X) -> az.InferenceData` - Generate posterior predictions for new data
- `score(X, y) -> pd.Series` - Compute R² scores (mean and std across posterior)
- `calculate_impact(y_true, y_pred) -> xr.DataArray` - Compute causal impact (y_true - y_pred)
- `calculate_cumulative_impact(impact) -> xr.DataArray` - Cumulative impact over time
- `print_coefficients(labels, round_to)` - Pretty-print posterior mean coefficients

**Data Format:**
- Expects `X` and `y` as `xarray.DataArray` objects
- Required coordinates:
  - `"obs_ind"` - Observation index
  - `"coeffs"` - Coefficient names (for X)
  - `"treated_units"` - Unit identifiers (for y, can be single unit)

**Concrete Implementations:**
1. `LinearRegression` - Standard Bayesian linear regression
2. `WeightedSumFitter` - For synthetic control (weighted combination of control units)
3. `InstrumentalVariableRegression` - Two-stage IV regression
4. `PropensityScore` - Logistic regression for propensity score estimation

### ScikitLearnAdaptor and sklearn Integration (`causalpy/skl_models.py`)

**ScikitLearnAdaptor** is a mixin class that adds CausalPy-specific methods to sklearn regressors.

**Key Methods:**
- `calculate_impact(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray` - Compute y_true - y_pred
- `calculate_cumulative_impact(impact: np.ndarray) -> np.ndarray` - Cumulative sum of impact
- `print_coefficients(labels: list[str], round_to: int | None)` - Pretty-print coefficients
- `get_coeffs() -> np.ndarray` - Extract coefficients as numpy array

**create_causalpy_compatible_class(estimator: RegressorMixin) -> RegressorMixin**

This function dynamically adds ScikitLearnAdaptor methods to any sklearn regressor instance. It's called automatically in `BaseExperiment.__init__()` when a sklearn model is provided.

**Built-in CausalPy sklearn Models:**
- `WeightedProportion` - Constrained regression for synthetic control (weights sum to 1, non-negative)

**Data Format:**
- Works with standard numpy arrays
- No special coordinate requirements

---

## 4. Public API Surface

### Explicitly Exported (in `causalpy/__init__.py` → `__all__`)

**Experiment Classes:**
- `DifferenceInDifferences` - Difference-in-differences analysis
- `InterruptedTimeSeries` - Interrupted time series analysis
- `RegressionDiscontinuity` - Regression discontinuity design
- `RegressionKink` - Regression kink design
- `SyntheticControl` - Synthetic control method
- `InstrumentalVariable` - Instrumental variable regression
- `InversePropensityWeighting` - Inverse propensity score weighting
- `PrePostNEGD` - Pre-post non-equivalent group design

**Model Modules:**
- `pymc_models` - Module containing all PyMC model classes
- `skl_models` - Module containing sklearn adaptors

**Helper Functions:**
- `create_causalpy_compatible_class(estimator)` - Make sklearn models CausalPy-compatible
- `load_data(dataset)` - Load example datasets

**Metadata:**
- `__version__` - Package version string

### Semi-Public API (used in examples, not in `__all__`)

**PyMC Model Classes** (access via `cp.pymc_models.ClassName`):
- `cp.pymc_models.LinearRegression()`
- `cp.pymc_models.WeightedSumFitter()`
- `cp.pymc_models.InstrumentalVariableRegression()`
- `cp.pymc_models.PropensityScore()`
- `cp.pymc_models.PyMCModel` (base class)

**Sklearn Models** (access via `cp.skl_models.ClassName`):
- `cp.skl_models.WeightedProportion()`

**Example Usage Pattern:**
```python
import causalpy as cp
from sklearn.linear_model import LinearRegression

# Use PyMC model
result = cp.RegressionDiscontinuity(
    df,
    formula="y ~ 1 + x + treated",
    model=cp.pymc_models.LinearRegression(),
    ...
)

# Use sklearn model
sklearn_model = LinearRegression()
result = cp.RegressionDiscontinuity(
    df,
    formula="y ~ 1 + x + treated",
    model=sklearn_model,  # Automatically wrapped
    ...
)
```

### Internal API (subject to change without notice)

**Not Intended for Direct Use:**
- `BaseExperiment` - Base class (use concrete experiment classes instead)
- `causalpy.plot_utils.*` - Internal plotting helpers
- `causalpy.reporting.*` - Internal reporting functions
- `causalpy.utils.*` - Internal utilities
- `causalpy.custom_exceptions.*` - Import if needed, but structure may change
- Abstract methods like `_bayesian_plot()`, `_ols_plot()` (use `plot()` instead)

**Import Guideline:**
- ✅ **Good**: `import causalpy as cp; cp.RegressionDiscontinuity(...)`
- ✅ **Good**: `from causalpy import RegressionDiscontinuity`
- ⚠️ **Discouraged**: `from causalpy.experiments.base import BaseExperiment`
- ❌ **Bad**: `from causalpy.plot_utils import plot_xY` (internal)

---

## 5. Workflows and Usage Patterns

Each quasi-experimental design follows a similar workflow but has design-specific requirements.

### Common Workflow Pattern

```python
import causalpy as cp

# 1. Load/prepare data
df = cp.load_data("dataset_name")
# or: df = pd.read_csv("my_data.csv")

# 2. Instantiate experiment with data, formula, and model
result = cp.ExperimentClass(
    data=df,
    formula="y ~ 1 + x1 + x2",
    design_specific_param="value",
    model=cp.pymc_models.LinearRegression(),
)

# 3. Visualize results
fig, ax = result.plot()

# 4. Get statistical summary
summary = result.effect_summary()
print(summary.text)
print(summary.table)

# 5. Extract underlying data
plot_data = result.get_plot_data()
```

---

### 1. Regression Discontinuity

**Class:** `RegressionDiscontinuity`

**Key Parameters:**
- `data: pd.DataFrame` - Dataset
- `formula: str` - Model formula (should include treatment variable)
- `running_variable_name: str` - Name of the running variable (assignment variable)
- `treatment_threshold: float` - Cutoff value for treatment assignment
- `model: Union[PyMCModel, RegressorMixin]` - Statistical model
- `bandwidth: float | None` - Optional bandwidth for local regression

**Data Requirements:**
- DataFrame must include running variable and outcome
- Treatment assignment typically based on `running_variable > threshold`

**Example Notebook:** `docs/source/notebooks/rd_pymc.ipynb`, `rd_skl.ipynb`, `rd_pymc_drinking.ipynb`

**Example:**
```python
result = cp.RegressionDiscontinuity(
    df,
    formula="outcome ~ 1 + running_var + treated",
    running_variable_name="running_var",
    treatment_threshold=21,
    model=cp.pymc_models.LinearRegression(),
)
```

---

### 2. Difference in Differences

**Class:** `DifferenceInDifferences`

**Key Parameters:**
- `data: pd.DataFrame` - Dataset
- `formula: str` - Model formula (should include `group*post_treatment` interaction)
- `time_variable_name: str` - Name of time variable column
- `group_variable_name: str` - Name of group variable column
- `post_treatment_variable_name: str` - Name of post-treatment indicator (default: "post_treatment")
- `model: Union[PyMCModel, RegressorMixin]` - Statistical model

**Data Requirements:**
- Must have group identifier (treatment vs control)
- Must have time variable
- Must have post-treatment indicator (0/1)
- Formula should include interaction term: `group*post_treatment`

**Example Notebook:** `docs/source/notebooks/did_pymc.ipynb`, `did_skl.ipynb`, `did_pymc_banks.ipynb`

**Example:**
```python
result = cp.DifferenceInDifferences(
    df,
    formula="y ~ 1 + group*post_treatment",
    time_variable_name="time",
    group_variable_name="group",
    model=cp.pymc_models.LinearRegression(),
)
```

---

### 3. Interrupted Time Series

**Class:** `InterruptedTimeSeries`

**Key Parameters:**
- `data: pd.DataFrame` - Time series data
- `treatment_time: Union[int, float, pd.Timestamp]` - Time point of intervention
- `formula: str` - Model formula
- `model: Union[PyMCModel, RegressorMixin]` - Statistical model

**Data Requirements:**
- Time-ordered observations (index should be time variable)
- Observations before and after `treatment_time`
- Index must be named `"obs_ind"` or will be set automatically

**Example Notebook:** `docs/source/notebooks/its_pymc.ipynb`, `its_skl.ipynb`, `its_covid.ipynb`

**Example:**
```python
result = cp.InterruptedTimeSeries(
    df,
    treatment_time=100,
    formula="y ~ 1 + t",
    model=cp.pymc_models.LinearRegression(),
)
```

---

### 4. Synthetic Control

**Class:** `SyntheticControl`

**Key Parameters:**
- `data: pd.DataFrame` - Panel data (unit × time)
- `treatment_time: Union[int, float, pd.Timestamp]` - When treatment occurred
- `formula: str` - Model formula
- `model: Union[PyMCModel, RegressorMixin]` - Typically `WeightedSumFitter` or `WeightedProportion`

**Data Requirements:**
- Panel data with multiple units (one or more treated, multiple control)
- Unit identifier and time variable
- Pre-treatment period for all units to fit weights

**Example Notebook:** `docs/source/notebooks/sc_pymc.ipynb`, `sc_skl.ipynb`, `sc_pymc_brexit.ipynb`

**Example:**
```python
result = cp.SyntheticControl(
    df,
    treatment_time=2016,
    formula="outcome ~ 0 + unit",  # No intercept, unit fixed effects
    model=cp.pymc_models.WeightedSumFitter(),
)
```

---

### 5. Regression Kink

**Class:** `RegressionKink`

**Key Parameters:**
- `data: pd.DataFrame` - Dataset
- `formula: str` - Model formula
- `kink_point: float` - Location of the kink
- `model: Union[PyMCModel, RegressorMixin]` - Statistical model
- `running_variable_name: str` - Name of the running variable

**Data Requirements:**
- Similar to RD but identifies **slope changes** rather than level changes
- Running variable with kink point where slope changes

**Example Notebook:** `docs/source/notebooks/rkink_pymc.ipynb`

**Example:**
```python
result = cp.RegressionKink(
    df,
    formula="y ~ 1 + x",
    kink_point=50,
    running_variable_name="x",
    model=cp.pymc_models.LinearRegression(),
)
```

---

### 6. Instrumental Variable

**Class:** `InstrumentalVariable`

**Key Parameters:**
- `data: pd.DataFrame` - Dataset
- `formula: str` - Structural equation formula
- `instruments_formula: str` - First-stage formula with instruments
- `model: PyMCModel` - Must be `InstrumentalVariableRegression` for PyMC
- `priors: Dict` - Prior specifications

**Data Requirements:**
- Endogenous variable(s)
- Valid instrumental variable(s) - correlated with endogenous variable, uncorrelated with error
- Outcome variable

**Backend Support:** Currently only supports PyMC backend (`supports_bayes=True`, `supports_ols=False`)

**Example Notebook:** `docs/source/notebooks/iv_pymc.ipynb`, `iv_weak_instruments.ipynb`

**Example:**
```python
result = cp.InstrumentalVariable(
    df,
    formula="outcome ~ 1 + endogenous_var",
    instruments_formula="endogenous_var ~ 1 + instrument",
    model=cp.pymc_models.InstrumentalVariableRegression(),
)
```

---

### 7. Inverse Propensity Weighting

**Class:** `InversePropensityWeighting`

**Key Parameters:**
- `data: pd.DataFrame` - Dataset with treatment and covariates
- `formula: str` - Outcome model formula
- `model: Union[PyMCModel, RegressorMixin]` - Model for outcome
- `weighting_scheme: str` - "ips" (inverse propensity score) or other schemes

**Data Requirements:**
- Binary treatment indicator
- Covariates for propensity score estimation
- Outcome variable

**Example Notebook:** `docs/source/notebooks/inv_prop_pymc.ipynb`, `inv_prop_latent.ipynb`

**Example:**
```python
result = cp.InversePropensityWeighting(
    df,
    formula="outcome ~ 1 + treatment",
    model=cp.pymc_models.LinearRegression(),
    weighting_scheme="ips",
)
```

---

### 8. PrePostNEGD (ANCOVA)

**Class:** `PrePostNEGD`

**Key Parameters:**
- `data: pd.DataFrame` - Dataset
- `formula: str` - Model formula (typically includes baseline covariate)
- `group_variable_name: str` - Name of group variable
- `pretreatment_variable_name: str` - Name of baseline/pretreatment covariate
- `model: Union[PyMCModel, RegressorMixin]` - Statistical model

**Data Requirements:**
- Pre-treatment (baseline) measurement
- Post-treatment (outcome) measurement
- Group indicator (treatment vs control)

**Example Notebook:** `docs/source/notebooks/ancova_pymc.ipynb`

**Example:**
```python
result = cp.PrePostNEGD(
    df,
    formula="post ~ 1 + pre + group",
    group_variable_name="group",
    pretreatment_variable_name="pre",
    model=cp.pymc_models.LinearRegression(),
)
```

---

### Common Methods Across All Experiments

After instantiating any experiment class, the following methods are available:

**Visualization:**
```python
fig, ax = result.plot()  # Automatic backend dispatch
```

**Statistical Summary:**
```python
summary = result.effect_summary(
    window="post",           # Time window (ITS/SC only)
    direction="increase",    # Expected effect direction
    alpha=0.05,             # Significance level
    cumulative=True,        # Include cumulative effects
    relative=True,          # Include relative (%) effects
    min_effect=None,        # ROPE threshold (PyMC only)
    treated_unit=None,      # For multi-unit experiments
)
print(summary.text)   # Prose summary
print(summary.table)  # Statistics table
```

**Extract Data:**
```python
plot_data = result.get_plot_data()  # Returns pd.DataFrame with predictions, CIs, etc.
```

**Model Coefficients:**
```python
result.print_coefficients(round_to=3)
```

**Access InferenceData (PyMC only):**
```python
idata = result.idata  # Access full posterior samples
```

---

## 6. Backends and Dependencies

### PyMC Backend (Bayesian Inference)

**Purpose:** Full Bayesian inference via MCMC sampling

**Key Features:**
- Posterior distributions for all parameters
- HDI (Highest Density Interval) credible intervals
- Probabilistic statements: P(effect > 0 | data)
- ROPE (Region of Practical Equivalence) analysis
- Posterior predictive distributions

**Data Structures:**
- **Input:** `xarray.DataArray` objects with named coordinates
  - Required coords: `"obs_ind"`, `"coeffs"`, `"treated_units"`
- **Output:** `arviz.InferenceData` objects
  - Contains posterior, prior, posterior_predictive, observed_data groups

**Model Building:**
All PyMC models inherit from `PyMCModel` and implement:
```python
def build_model(self, X, y, coords):
    with self:
        # Define priors
        # Define likelihood
        # Define observed data
```

**Sampling:**
Models are fit via `fit()` which calls `pm.sample()`:
```python
model = cp.pymc_models.LinearRegression(
    sample_kwargs={
        "draws": 2000,
        "chains": 4,
        "random_seed": 42,
        "target_accept": 0.95,
    }
)
```

**Available PyMC Models:**
- `LinearRegression` - Standard linear regression
- `WeightedSumFitter` - Weighted combination (synthetic control)
- `InstrumentalVariableRegression` - Two-stage IV
- `PropensityScore` - Logistic regression for propensity scores

**Dependencies:**
- `pymc >= 5.15.1`
- `arviz >= 0.14.0`
- `xarray >= 2022.11.0`
- `pytensor` (via pymc)
- `pymc-extras >= 0.3.0`

---

### Scikit-learn Backend (Frequentist/OLS)

**Purpose:** Fast point estimates with asymptotic confidence intervals

**Key Features:**
- Point estimates and standard errors
- Asymptotic confidence intervals (based on t-distribution)
- Much faster than MCMC
- Easy to use any sklearn regressor

**Data Structures:**
- **Input:** `numpy.ndarray` (standard sklearn format)
- **Output:** Fitted model with `coef_`, `predict()`, etc.

**Model Compatibility:**
Any sklearn regressor can be used. CausalPy automatically adds required methods via `create_causalpy_compatible_class()`:
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# All of these work:
model = LinearRegression()
model = Ridge(alpha=1.0)
model = Lasso(alpha=0.1)
model = RandomForestRegressor()
```

**Built-in CausalPy sklearn Models:**
- `WeightedProportion` - Constrained weights for synthetic control
  - Constraints: weights sum to 1, all weights ≥ 0
  - Uses `scipy.optimize.fmin_slsqp` for constrained optimization

**Dependencies:**
- `scikit-learn >= 1.0`
- `numpy`
- `scipy`

---

### Key Dependencies Summary

**Core Statistical:**
- `pymc >= 5.15.1` - Bayesian modeling
- `scikit-learn >= 1.0` - OLS modeling
- `arviz >= 0.14.0` - Bayesian inference diagnostics
- `xarray >= 2022.11.0` - Labeled arrays
- `statsmodels` - Additional statistical functions

**Data and Formulas:**
- `pandas` - DataFrames
- `numpy` - Numerical arrays
- `patsy` - Formula parsing (R-style formulas)

**Visualization:**
- `matplotlib >= 3.5.3` - Plotting
- `seaborn >= 0.11.2` - Statistical plotting
- `graphviz` - DAG visualization

**Development:**
- `pytest` - Testing framework
- `ruff` - Linting
- `mypy` - Type checking
- `interrogate` - Docstring coverage
- `pre-commit` - Git hooks

See `pyproject.toml` for complete dependency list with version constraints.

---

## 7. Extension Points

### Adding a New Experiment Design

To add a new quasi-experimental design (e.g., "Comparative Interrupted Time Series"):

**1. Create a new file in `causalpy/experiments/`**

```python
# causalpy/experiments/my_new_design.py

from typing import Union
import pandas as pd
from sklearn.base import RegressorMixin
from causalpy.pymc_models import PyMCModel
from .base import BaseExperiment

class MyNewDesign(BaseExperiment):
    """
    Brief description of the design.

    Parameters
    ----------
    data : pd.DataFrame
        Description
    formula : str
        Model formula
    design_specific_param : str
        Design-specific parameter
    model : PyMCModel or RegressorMixin
        Statistical model
    """

    # Declare backend support
    supports_bayes = True  # Set to True if PyMC works
    supports_ols = True    # Set to True if sklearn works

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        design_specific_param: str,
        model: Union[PyMCModel, RegressorMixin],
    ):
        # Store parameters
        self.data = data
        self.formula = formula
        self.design_specific_param = design_specific_param

        # Call parent constructor (validates model compatibility)
        super().__init__(model=model)

        # Fit the model
        self._fit()

    def _fit(self):
        """Internal fit method"""
        # Parse formula, validate data, fit model
        pass

    # Implement abstract methods for supported backends:

    def _bayesian_plot(self, *args, **kwargs):
        """Plotting for PyMC models"""
        if not self.supports_bayes:
            raise NotImplementedError("Bayesian models not supported")
        # Create matplotlib figure and axes
        # Return (fig, ax) tuple
        pass

    def _ols_plot(self, *args, **kwargs):
        """Plotting for sklearn models"""
        if not self.supports_ols:
            raise NotImplementedError("OLS models not supported")
        # Create matplotlib figure and axes
        # Return (fig, ax) tuple
        pass

    def get_plot_data_bayesian(self, *args, **kwargs) -> pd.DataFrame:
        """Extract plot data for PyMC"""
        if not self.supports_bayes:
            raise NotImplementedError("Bayesian models not supported")
        # Return DataFrame with predictions, HDIs, etc.
        pass

    def get_plot_data_ols(self, *args, **kwargs) -> pd.DataFrame:
        """Extract plot data for sklearn"""
        if not self.supports_ols:
            raise NotImplementedError("OLS models not supported")
        # Return DataFrame with predictions, CIs, etc.
        pass
```

**2. Export in `causalpy/experiments/__init__.py`**

```python
from .my_new_design import MyNewDesign

__all__ = [..., "MyNewDesign"]
```

**3. Add to main package `causalpy/__init__.py`**

```python
from .experiments.my_new_design import MyNewDesign

__all__ = [..., "MyNewDesign"]
```

**4. Write tests in `causalpy/tests/`**

Create or update test files:
- Unit tests for the class
- Integration test with PyMC model (if supported)
- Integration test with sklearn model (if supported)

**5. Add example notebook in `docs/source/notebooks/`**

- Name pattern: `{method}_{backend}.ipynb` (e.g., `mynewdesign_pymc.ipynb`)
- Show realistic example with interpretation
- Link to relevant literature

**6. Update documentation**

- Ensure docstrings follow NumPy style
- Add entry to README.md table of methods
- Update this RESEARCH.md file

---

### Adding a New PyMC Model

To add a new Bayesian model (e.g., "Bayesian Additive Regression Trees"):

**1. Add class to `causalpy/pymc_models.py`**

```python
class MyNewModel(PyMCModel):
    """
    Description of the model.

    Parameters
    ----------
    sample_kwargs : dict, optional
        Passed to pm.sample()
    priors : dict, optional
        Custom prior specifications
    """

    # Optional: Define default priors
    default_priors = {
        "beta": Prior("Normal", mu=0, sigma=1, dims=["treated_units", "coeffs"]),
        "sigma": Prior("HalfNormal", sigma=1, dims="treated_units"),
    }

    def priors_from_data(self, X, y):
        """
        Generate data-adaptive priors (optional).

        Returns
        -------
        dict
            Prior specifications based on data scale
        """
        y_std = float(y.std())
        return {
            "sigma": Prior("HalfNormal", sigma=y_std, dims="treated_units"),
        }

    def build_model(self, X, y, coords):
        """
        Build the PyMC model.

        Parameters
        ----------
        X : xr.DataArray
            Features with dims ["obs_ind", "coeffs"]
        y : xr.DataArray
            Outcome with dims ["obs_ind", "treated_units"]
        coords : dict
            Coordinate specifications
        """
        with self:
            self.add_coords(coords)

            # Add data containers
            X_ = pm.Data("X", X.values, dims=["obs_ind", "coeffs"])
            y_ = pm.Data("y", y.values, dims=["obs_ind", "treated_units"])

            # Define priors (use self._generate_and_set_priors())
            self._generate_and_set_priors(
                X, y, **self._merge_prior_dicts(X, y, self.user_priors)
            )

            # Define likelihood
            # ...

            # Observed data
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_, dims=["obs_ind", "treated_units"])
```

**2. Write tests**

Add tests to `causalpy/tests/test_pymc_models.py`:
- Test that model builds correctly
- Test fit/predict/score methods
- Test with different data shapes

**3. Add example usage**

Create or update notebook showing when to use this model.

---

### Adding a New scikit-learn Backend

Most sklearn regressors work out-of-the-box via `create_causalpy_compatible_class()`. However, you can create specialized sklearn models:

**1. Add class to `causalpy/skl_models.py`**

```python
class MySpecializedModel(ScikitLearnAdaptor, LinearModel, RegressorMixin):
    """
    Description of the model.

    Parameters
    ----------
    param1 : type
        Description
    """

    def __init__(self, param1=None):
        self.param1 = param1

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model on data X with outcome y"""
        # Fitting logic
        self.coef_ = ...  # Must set coef_ attribute
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes for data X"""
        return np.dot(X, self.coef_.T)
```

**2. Write tests**

Add tests to `causalpy/tests/test_integration_skl_examples.py`

---

### Naming Conventions

**Classes:**
- PascalCase: `DifferenceInDifferences`, `LinearRegression`
- Descriptive names that reflect the method

**Methods:**
- snake_case: `fit()`, `calculate_impact()`, `print_coefficients()`
- Follow sklearn patterns: `fit()`, `predict()`, `score()`

**Internal/Private:**
- Prefix with underscore: `_bayesian_plot()`, `_fit()`, `_validate_data()`

**Files:**
- snake_case: `diff_in_diff.py`, `pymc_models.py`

---

## 8. Testing and Quality

### Test Organization (`causalpy/tests/`)

**Unit Tests:**
- `test_pymc_models.py` - PyMC model classes
- `test_utils.py` - Utility functions
- `test_plot_utils.py` - Plotting helpers
- `test_reporting.py` - Reporting functions
- `test_data_loading.py` - Dataset loading

**Integration Tests:**
- `test_integration_pymc_examples.py` - Full workflows with PyMC backend
- `test_integration_skl_examples.py` - Full workflows with sklearn backend

**Validation Tests:**
- `test_input_validation.py` - Data validation and error messages
- `test_model_experiment_compatability.py` - Model/experiment compatibility checks

**Other:**
- `test_misc.py` - Miscellaneous functionality
- `test_synthetic_data.py` - Synthetic data generation
- `conftest.py` - Shared pytest fixtures

### Test Conventions

**Style:**
```python
# ✅ Good: pytest-style function
def test_linear_regression_fit():
    model = LinearRegression()
    result = model.fit(X, y)
    assert result is not None

# ❌ Bad: unittest-style class
class TestLinearRegression(unittest.TestCase):
    def test_fit(self):
        ...
```

**Use Fixtures:**
```python
@pytest.fixture
def sample_data():
    return pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})

def test_with_fixture(sample_data):
    assert len(sample_data) == 3
```

**Minimize MCMC Sampling:**
```python
# For PyMC tests, use minimal sampling to keep tests fast
model = cp.pymc_models.LinearRegression(
    sample_kwargs={
        "draws": 100,      # Minimal draws
        "chains": 2,       # Minimal chains
        "progressbar": False,
        "random_seed": 42,
    }
)
```

**Test Markers:**
```python
@pytest.mark.integration
def test_full_workflow():
    ...

@pytest.mark.slow
def test_expensive_computation():
    ...
```

### Running Tests

**All tests:**
```bash
make test
# or: pytest causalpy/tests/
```

**Specific test file:**
```bash
pytest causalpy/tests/test_pymc_models.py
```

**Specific test function:**
```bash
pytest causalpy/tests/test_pymc_models.py::test_linear_regression_fit
```

**With coverage:**
```bash
pytest --cov=causalpy --cov-report=html
```

**Integration tests only:**
```bash
pytest -m integration
```

### Quality Tools

**MyPy (Type Checking):**
```bash
mypy causalpy/
# Configured in pyproject.toml [tool.mypy]
# Runs on pre-commit hook
```

**Ruff (Linting):**
```bash
make check_lint  # Check for errors
make lint        # Auto-fix errors
# Configured in pyproject.toml [tool.ruff]
```

**Interrogate (Docstring Coverage):**
```bash
interrogate causalpy/
# Minimum coverage: 85% (configured in pyproject.toml)
# Badge generated at docs/source/_static/interrogate_badge.svg
```

**Codespell (Spell Checking):**
```bash
codespell causalpy/
# Configured in pyproject.toml [tool.codespell]
```

**Pre-commit Hooks:**
```bash
pre-commit install  # Install hooks
pre-commit run --all-files  # Run all hooks manually
```

Hooks include: ruff, mypy, codespell, trailing whitespace removal, etc.

**Coverage (Codecov):**
- Automated via GitHub Actions
- Reports uploaded to https://codecov.io/gh/pymc-labs/CausalPy
- Configured in `codecov.yml`

### Documentation Testing

**Doctest (examples in docstrings):**
```bash
make doctest
# or: pytest --doctest-modules causalpy/
```

Test specific module:
```bash
pytest --doctest-modules causalpy/pymc_models.py
```

---

## 9. Known Limitations and Open Questions

### Current Limitations

**1. Data Structure Heterogeneity**
- PyMC models expect `xarray.DataArray` with specific coordinates
- Sklearn models expect `numpy.ndarray`
- Conversion logic scattered across experiment classes
- **Impact:** Makes code harder to maintain and test
- **Potential solution:** Unified data container that adapts to backend

**2. Backend Support Asymmetry**
- Not all experiments support both backends
- Example: `InstrumentalVariable` only supports PyMC (`supports_ols=False`)
- **Impact:** Users may be surprised by backend limitations
- **Potential solution:** Clear documentation, consider adding sklearn IV support

**3. Index Naming Requirement**
- Data index must be named `"obs_ind"` for consistency
- Experiments may automatically set this, but it's not always explicit
- **Impact:** Can cause confusion when data has other index names
- **Potential solution:** Better error messages, automatic handling

**4. Formula Parsing Dependency**
- Relies on `patsy` library for R-style formulas
- Patsy is mature but not actively developed
- **Open question:** Should we migrate to `formulaic`?
- **Impact:** Long-term maintenance concern

**5. Plotting Assumptions**
- Plotting methods have many design-specific assumptions
- Difficult to customize without understanding internals
- **Impact:** Limited flexibility for custom visualizations
- **Potential solution:** Expose more plotting parameters, better plot data extraction

**6. Effect Summary Complexity**
- `effect_summary()` has grown complex with many parameters
- Different experiments use subsets of parameters
- **Impact:** API can be confusing
- **Potential solution:** Design-specific summary methods?

### Architectural Debt

**1. Tight Coupling**
- Experiment classes directly instantiate plotting code
- Hard to swap out visualization library
- **Potential solution:** Abstract plotting interface

**2. Validation Scattered**
- Data validation spread across multiple methods
- Some in `__init__`, some in `fit()`
- **Potential solution:** Centralized validation utilities

**3. Reporting Module Size**
- `reporting.py` has grown to >1400 lines
- Contains many helper functions
- **Potential solution:** Split into submodules by experiment type

### Performance Considerations

**1. MCMC Sampling Speed**
- PyMC models can be slow for large datasets
- No automatic suggestions for sampling parameters
- **Open question:** Should we provide adaptive sampling configurations?

**2. Memory Usage**
- Posterior samples can consume significant memory for large models
- No built-in thinning or compression
- **Impact:** May hit memory limits on large problems

### Design Questions

**1. Should we support other backends?**
- TensorFlow Probability?
- Stan?
- NumPyro/JAX?
- **Trade-off:** Flexibility vs maintenance burden

**2. Should experiments be classes or functions?**
- Current: Class-based with state
- Alternative: Functional API returning result objects
- **Trade-off:** Flexibility vs simplicity

**3. How to handle causal assumptions?**
- Currently implicit in design choice
- Should we have explicit assumption checking?
- Should we visualize assumptions (DAGs)?
- **Open question:** How much causal inference theory to encode?

### Areas Needing Refactor

**1. Coordinate handling**
- xarray coordinate management is verbose and error-prone
- Many repeated patterns across PyMC models
- **Candidate:** Extract coordinate utilities

**2. Prior specification**
- Prior handling improved with `pymc-extras.Prior`, but still complex
- Default priors may not be sensible for all scales
- **Candidate:** More robust data-adaptive priors

**3. Model selection/comparison**
- No built-in tools for comparing different models
- Users must manually compare WAIC/LOO
- **Candidate:** Add model comparison utilities

---

## 10. Update Policy

### When to Update RESEARCH.md

**Must Update:**
- ✅ Adding or removing experiment types (Section 5)
- ✅ Adding or removing model classes (Sections 3, 6)
- ✅ Changing public API (Section 4)
- ✅ Major architectural refactors (Sections 2, 3)
- ✅ Adding new backends (Section 6)
- ✅ Changing extension patterns (Section 7)

**Should Update:**
- ⚠️ Discovering new limitations (Section 9)
- ⚠️ Changing test organization (Section 8)
- ⚠️ Updating dependencies (Section 6)
- ⚠️ Adding major features to existing classes (Sections 3, 5)

**Optional Update:**
- ℹ️ Minor method additions
- ℹ️ Bug fixes (unless they reveal architectural issues)
- ℹ️ Documentation improvements (update if architecture changes)

### Section Update Frequency

**Rarely Change (stable):**
- Section 1: Purpose and Scope
- Section 2: High-Level Architecture (structure)
- Section 3: Core Abstractions (base classes)
- Section 7: Extension Points (patterns)
- Section 10: Update Policy (meta)

**Occasionally Change:**
- Section 4: Public API Surface (when adding exports)
- Section 6: Backends and Dependencies (version bumps, new backends)
- Section 8: Testing and Quality (test reorganization)

**Frequently Change:**
- Section 5: Workflows and Usage Patterns (new notebooks, parameter changes)
- Section 9: Known Limitations (as we discover and fix issues)

### How to Update

**1. Update inline with code changes:**
```bash
# Make code changes
git add causalpy/experiments/new_design.py

# Update RESEARCH.md immediately
vim RESEARCH.md
git add RESEARCH.md

# Commit together
git commit -m "Add NewDesign experiment class

- Implement NewDesign for ... use case
- Add to public API exports
- Update RESEARCH.md Section 5 with usage pattern
"
```

**2. Review before major releases:**
- Check if all sections accurately reflect current code
- Update version number and last updated date at top
- Run through each section systematically

**3. Link to RESEARCH.md in PR templates:**
- Remind contributors to update when needed
- Include checklist item: "Updated RESEARCH.md if API/architecture changed"

### Validation

Before committing RESEARCH.md updates:

1. ✅ All class/module names are spelled correctly
2. ✅ All file paths exist and are accurate
3. ✅ Code examples are syntactically valid
4. ✅ Links to notebooks point to real files
5. ✅ Version numbers match `pyproject.toml`
6. ✅ New sections follow existing style and formatting

### Ownership

- **Primary maintainers:** Responsible for keeping architecture sections accurate
- **Contributors:** Should update when adding features
- **Reviewers:** Check RESEARCH.md updates in PRs

---

## Appendix: Quick Reference

### Key Files Map

```
causalpy/
├── __init__.py                    → Section 4 (Public API)
├── experiments/
│   ├── base.py                    → Section 3 (BaseExperiment)
│   └── *.py                       → Section 5 (Workflows)
├── pymc_models.py                 → Sections 3, 6 (PyMCModel)
├── skl_models.py                  → Sections 3, 6 (sklearn)
├── plot_utils.py                  → Section 2 (utilities)
├── reporting.py                   → Section 2 (utilities)
└── tests/                         → Section 8 (Testing)
```

### Import Cheatsheet

```python
# Experiments (all public)
from causalpy import (
    DifferenceInDifferences,
    InterruptedTimeSeries,
    RegressionDiscontinuity,
    RegressionKink,
    SyntheticControl,
    InstrumentalVariable,
    InversePropensityWeighting,
    PrePostNEGD,
)

# PyMC models (semi-public)
import causalpy as cp
model = cp.pymc_models.LinearRegression()
model = cp.pymc_models.WeightedSumFitter()
model = cp.pymc_models.InstrumentalVariableRegression()
model = cp.pymc_models.PropensityScore()

# Sklearn models
model = cp.skl_models.WeightedProportion()

# Utilities (public)
from causalpy import load_data, create_causalpy_compatible_class

# Sklearn compatibility
from sklearn.linear_model import Ridge
model = Ridge()  # Automatically wrapped by experiment class
```

### Common Gotchas

1. **Forgot to set `supports_bayes`/`supports_ols`** → Model compatibility error
2. **Index not named "obs_ind"** → Coordinate mismatch errors
3. **Missing interaction term in DiD formula** → Incorrect causal effect estimate
4. **Using `pm.Model` instead of `PyMCModel`** → Missing required methods
5. **Not calling `create_causalpy_compatible_class()`** → sklearn model lacks `calculate_impact()`

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-18 | Initial comprehensive RESEARCH.md created |

---

**For questions or suggestions about this document, open an issue on GitHub.**
