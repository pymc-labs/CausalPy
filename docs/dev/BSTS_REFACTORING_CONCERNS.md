# BSTS Implementation: API Conformance Issues and Refactoring Recommendations

## Overview

The BSTS (Bayesian Structural Time Series) feature branch adds two new model classes (`BayesianBasisExpansionTimeSeries` and `StateSpaceTimeSeries`) and modifies the `InterruptedTimeSeries` experiment class to support them. While the implementation is functional, there are significant deviations from the established patterns in CausalPy that reduce maintainability and violate key design principles.

This document outlines the major concerns and proposes solutions to align the BSTS implementation with CausalPy's architecture.

---

## 🚨 Critical Issues

### 1. API Inconsistency - Data Type Signatures (`pymc_models.py`)

**Problem:**
The new model classes break the established contract that all `PyMCModel` subclasses accept `xr.DataArray`:

```python
# Existing pattern (all other models)
def build_model(self, X: xr.DataArray, y: xr.DataArray, coords: Dict[str, Any] | None)
def fit(self, X: xr.DataArray, y: xr.DataArray, coords: Dict[str, Any] | None)

# New BSTS models
def build_model(self, X: Optional[np.ndarray], y: np.ndarray, coords: Dict[str, Any] | None)
def fit(self, X: Optional[np.ndarray], y: np.ndarray, coords: Dict[str, Any] | None)
```

**Impact:**
- Violates Liskov Substitution Principle
- Forces experiment classes to use `isinstance()` checks and data conversions
- Makes the API unpredictable for users
- Breaks polymorphism

**Evidence:**
- `interrupted_time_series.py:163-164`: Complex data conversion logic
- `interrupted_time_series.py:157-158, 185-186, 204-205, 222-223, 246-247`: Five repeated type checks

---

### 2. Missing `treated_units` Dimension (`pymc_models.py`)

**Problem:**
BSTS models omit the `treated_units` dimension that all other models include:

```python
# Existing pattern
mu = pm.Deterministic("mu", ..., dims=["obs_ind", "treated_units"])

# New BSTS models
mu = pm.Deterministic("mu", mu_, dims="obs_ind")  # Missing treated_units!
```

**Impact:**
- Breaks the base class `score()` method (line 333 expects `treated_units`)
- Breaks the base class `_data_setter()` (lines 220-223 expect `treated_units`)
- Forces complete override of `score()` in both model classes
- Requires defensive checks throughout experiment plotting code

**Evidence:**
- `pymc_models.py:1412, 1417`: BSTS models use `dims="obs_ind"` only
- `interrupted_time_series.py:319-321, 344-348, 369-371`: ~15 conditional checks for `treated_units` in plotting
- `interrupted_time_series.py:407-410, 432-433, 436-439`: ~8 `hasattr` checks in data extraction

---

### 3. Return Type Inconsistency (`pymc_models.py`)

**Problem:**
`StateSpaceTimeSeries.predict()` returns `xr.Dataset` instead of `az.InferenceData`:

```python
# Base class contract
def predict(self, X: xr.DataArray, ...) -> az.InferenceData

# StateSpaceTimeSeries violation
def predict(self, X: Optional[np.ndarray], ...) -> xr.Dataset  # Line 1811
```

**Impact:**
- Breaks polymorphism
- Requires defensive wrapping in experiment class (lines 213-214, 235-238)
- Users can't reliably use `.predict()` without checking instance types

**Evidence:**
```python
# interrupted_time_series.py:213-214, 235-238
if not isinstance(self.pre_pred, az.InferenceData):
    self.pre_pred = az.InferenceData(posterior_predictive=self.pre_pred)
```

---

### 4. Code Duplication - Repeated Type Checks (`interrupted_time_series.py`)

**Problem:**
The same `isinstance()` check is repeated **5 times** in `__init__`:

```python
# Lines 157-158, 185-186, 204-205, 222-223, 246-247
is_bsts_like = isinstance(
    self.model, (BayesianBasisExpansionTimeSeries, StateSpaceTimeSeries)
)
```

**Impact:**
- Violates DRY (Don't Repeat Yourself) principle
- Creates maintenance burden - changes require updating 5 places
- Makes code harder to read and follow

**Comparison:**
Other experiment classes (DifferenceInDifferences, SyntheticControl, PrePostNEGD) do ONE type check:
```python
if isinstance(self.model, PyMCModel):
    # PyMC logic
elif isinstance(self.model, RegressorMixin):
    # SKL logic
```

---

### 5. Violation of Open/Closed Principle (`interrupted_time_series.py`)

**Problem:**
The experiment class imports and explicitly checks for specific model types:

```python
from causalpy.pymc_models import (
    BayesianBasisExpansionTimeSeries,  # ← Tight coupling
    PyMCModel,
    StateSpaceTimeSeries,  # ← Tight coupling
)
```

**Impact:**
- Adding new time-series models requires modifying the experiment class
- Breaks the abstraction provided by the `PyMCModel` base class
- Violates Open/Closed Principle (open for extension, closed for modification)

**Comparison:**
Other experiment files only import base classes:
```python
# diff_in_diff.py, synthetic_control.py, etc.
from causalpy.pymc_models import PyMCModel
```

---

## ⚠️ Major Issues

### 6. Special Coordinate Requirements (`pymc_models.py`)

**Problem:**
BSTS models require `datetime_index` as `pd.DatetimeIndex` in coords, and pop it from the dictionary:

```python
# Line 1281 (BayesianBasisExpansionTimeSeries)
datetime_index = coords.pop("datetime_index", None)
```

**Impact:**
- Makes API less predictable
- `datetime_index` is not preserved in model coordinates
- Users must know special requirements for these models

**Standard Pattern:**
```python
# Standard coords
{"coeffs": [...], "obs_ind": [...], "treated_units": [...]}
```

---

### 7. Non-Standard Model Context (`pymc_models.py`)

**Problem:**
`StateSpaceTimeSeries` creates a separate model context instead of using `self`:

```python
# Existing pattern
with self:  # Use the PyMCModel instance as context
    self.add_coords(coords)
    # ... model definition

# StateSpaceTimeSeries (Line 1717-1736)
with pm.Model(coords=coordinates) as self.second_model:
    # ... model definition
```

**Impact:**
- Confusing because `StateSpaceTimeSeries` inherits from `pm.Model`
- Breaks Liskov Substitution Principle
- Methods expecting `with self:` won't work correctly
- Creates maintenance complexity

---

### 8. No Prior Configuration System (`pymc_models.py`)

**Problem:**
BSTS models don't use the standard `default_priors` system:

```python
# Existing pattern
default_priors = {
    "beta": Prior("Normal", mu=0, sigma=50, dims=["treated_units", "coeffs"]),
    ...
}

# BSTS models - hard-coded priors
beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")  # Line 1408
sigma = pm.HalfNormal("sigma", sigma=self.prior_sigma)   # Line 1415
```

**Impact:**
- Users can't customize priors using the standard Prior system
- Only `prior_sigma` is configurable via `__init__`
- Inconsistent with established patterns

---

### 9. Complex `_data_setter()` Override (`pymc_models.py`)

**Problem:**
`BayesianBasisExpansionTimeSeries._data_setter()` has a different signature:

```python
# Base class
def _data_setter(self, X: xr.DataArray) -> None

# BayesianBasisExpansionTimeSeries (Line 1456-1536)
def _data_setter(self, X_pred: Optional[np.ndarray], coords_pred: Dict[str, Any]) -> None
```

**Impact:**
- Signature doesn't match base class
- Base `predict()` can't call it correctly
- Forces complete override of `predict()`

---

### 10. Extensive Conditional Logic in Plotting (`interrupted_time_series.py`)

**Problem:**
Plotting methods have ~15 conditional checks for `treated_units` dimension:

```python
# Lines 319-321, 344-348, 369-371, etc.
pre_mu_plot = (
    pre_mu.isel(treated_units=0) if "treated_units" in pre_mu.dims else pre_mu
)
```

**Impact:**
- Makes plotting code verbose and hard to read
- Other plotting methods don't need this complexity
- Suggests data format should be standardized earlier

---

### 11. Inconsistent Data Handling Pattern (`interrupted_time_series.py`)

**Problem:**
Experiment stores data as xarray, then converts to numpy for BSTS:

```python
# Lines 163-164
X_fit = self.pre_X.values if self.pre_X.shape[1] > 0 else None
y_fit = self.pre_y.isel(treated_units=0).values
```

**Impact:**
- Data stored in one format but used in another
- Conversion logic is complex and error-prone
- Complex conditional: `if self.pre_X.shape[1] > 0 else None`

**Standard Pattern:**
```python
# synthetic_control.py, lines 152-156
self.model.fit(
    X=self.datapre_control,  # ← xarray passed directly
    y=self.datapre_treated,
    coords=COORDS,
)
```

---

### 12. State Management Complexity (`pymc_models.py`)

**Problem:**
`BayesianBasisExpansionTimeSeries` maintains hidden state:

```python
# Line 1110, 1111
self._first_fit_timestamp: Optional[pd.Timestamp] = None
self._exog_var_names: Optional[List[str]] = None

# Line 1247
if self._first_fit_timestamp is None:
    self._first_fit_timestamp = datetime_index[0]
```

**Impact:**
- Makes model stateful in non-obvious ways
- First call to `fit()` permanently sets `_first_fit_timestamp`
- Subsequent predictions use this for time calculations
- No clear way to reset the model

---

## 🔧 Proposed Solutions

### Solution 1: Create `TimeSeriesPyMCModel` Abstract Base Class

**Approach:**
Create a new abstract base class that handles time-series-specific requirements:

```python
class TimeSeriesPyMCModel(PyMCModel):
    """Base class for time series models with datetime indices."""

    def build_model(
        self,
        X: Optional[np.ndarray],
        y: np.ndarray,
        coords: Dict[str, Any]
    ) -> None:
        """
        Time series models use numpy arrays and require datetime_index in coords.

        Parameters
        ----------
        X : np.ndarray or None
            Exogenous variables
        y : np.ndarray
            Target variable (1D)
        coords : dict
            Must contain "datetime_index" (pd.DatetimeIndex)
        """
        raise NotImplementedError

    def fit(
        self,
        X: Optional[np.ndarray],
        y: np.ndarray,
        coords: Dict[str, Any]
    ) -> az.InferenceData:
        """Fit time series model."""
        raise NotImplementedError

    # Add time-series specific helper methods
    def _validate_datetime_index(self, coords: Dict[str, Any]) -> pd.DatetimeIndex:
        """Extract and validate datetime index from coords."""
        ...
```

**Benefits:**
- Clear separation between standard and time-series models
- Experiment classes can use `isinstance(model, TimeSeriesPyMCModel)` once
- Documents the different requirements
- Allows future time-series models to extend easily

---

### Solution 2: Add `treated_units` Dimension to BSTS Models

**Approach:**
Modify BSTS models to always include `treated_units=["unit_0"]`:

```python
# In build_model()
model_coords = {
    "obs_ind": np.arange(num_obs),
    "treated_units": ["unit_0"],  # ← Add this
}

# Update mu definition
mu = pm.Deterministic("mu", mu_, dims=["obs_ind", "treated_units"])  # ← Add treated_units
```

**Benefits:**
- Maintains consistency with other models
- Base class methods work without modification
- Eliminates ~23 conditional checks in experiment class
- Simpler plotting code

**Trade-offs:**
- Slightly more complex for truly univariate models
- But improves overall consistency

---

### Solution 3: Standardize Return Types

**Approach:**
Make `StateSpaceTimeSeries.predict()` return `az.InferenceData`:

```python
def predict(self, ...) -> az.InferenceData:
    # ... existing logic ...

    # Wrap result in InferenceData before returning
    result = az.InferenceData(posterior_predictive={
        "y_hat": y_hat_final,
        "mu": y_hat_final,
    })
    return result
```

**Benefits:**
- Maintains polymorphism
- No defensive wrapping needed in experiment class
- Users can rely on consistent API

---

### Solution 4: Refactor Experiment Class to Reduce Duplication

**Approach:**
Extract repeated logic into helper methods:

```python
class InterruptedTimeSeries(BaseExperiment):
    def __init__(self, ...):
        super().__init__(model=model)
        # ... setup ...

        # Single type check
        self._is_timeseries_model = isinstance(
            self.model, TimeSeriesPyMCModel  # Or use ABC
        )

        # Extract to methods
        self._fit_model()
        self._score_model()
        self._predict_pre_period()
        self._predict_post_period()
        self._calculate_impacts()

    def _prepare_data_for_model(self, X: xr.DataArray, y: xr.DataArray):
        """Handle data format conversion in one place."""
        if self._is_timeseries_model:
            return self._convert_to_timeseries_format(X, y)
        return X, y

    def _convert_to_timeseries_format(self, X, y):
        """Convert xarray to format expected by time series models."""
        X_numpy = X.values if X.shape[1] > 0 else None
        y_numpy = y.isel(treated_units=0).values
        return X_numpy, y_numpy
```

**Benefits:**
- Reduces duplication from 5 checks to 1
- Centralizes conversion logic
- Easier to test
- More maintainable

---

### Solution 5: Implement Standard Prior System

**Approach:**
Add `default_priors` to BSTS models:

```python
class BayesianBasisExpansionTimeSeries(PyMCModel):
    default_priors = {
        "beta": Prior("Normal", mu=0, sigma=10, dims="coeffs"),
        "sigma": Prior("HalfNormal", sigma=5),
    }

    def __init__(self, ..., priors: dict[str, Any] | None = None):
        super().__init__(sample_kwargs=sample_kwargs, priors=priors)
        # ... rest of init ...

    def build_model(self, ...):
        # Use self.priors instead of hard-coded values
        beta = self.priors["beta"].create_variable("beta")
        sigma = self.priors["sigma"].create_variable("sigma")
```

**Benefits:**
- Users can customize priors using standard system
- Consistent with other models
- Better defaults documented in one place

---

### Solution 6: Add Helper Method for Model Context

**Approach:**
For `StateSpaceTimeSeries`, document why separate context is needed:

```python
class StateSpaceTimeSeries(PyMCModel):
    """
    Note: This model uses a separate PyMC Model context (self.second_model)
    instead of self due to requirements of the state-space implementation.
    This is necessary for pymc-extras state-space models.
    """

    def build_model(self, ...):
        # Current approach, but with clear documentation
        with pm.Model(coords=coordinates) as self.second_model:
            ...
```

Or if possible, refactor to use `self`:

```python
def build_model(self, ...):
    with self:
        self.add_coords(coordinates)
        # ... build state-space model within self context
```

---

## 📋 Implementation Plan

### Phase 1: Quick Wins (Low Risk, High Impact)
1. ✅ **Add experimental warnings** (DONE)
2. Extract repeated type check in `InterruptedTimeSeries.__init__` to single variable
3. Add `treated_units` dimension to BSTS models
4. Standardize `StateSpaceTimeSeries.predict()` return type

### Phase 2: API Standardization (Medium Risk, High Impact)
5. Create `TimeSeriesPyMCModel` abstract base class
6. Refactor BSTS models to inherit from new base class
7. Implement standard prior system in BSTS models
8. Update experiment class to use ABC instead of explicit type checks

### Phase 3: Code Quality (Low Risk, Medium Impact)
9. Extract helper methods in `InterruptedTimeSeries` to reduce duplication
10. Simplify plotting code (benefits from Phase 1 #3)
11. Add comprehensive documentation about time-series model requirements
12. Add tests for time-series model interface

### Phase 4: Advanced Improvements (Optional)
13. Consider adapter pattern to wrap BSTS models for xarray compatibility
14. Evaluate state management approach in `BayesianBasisExpansionTimeSeries`
15. Document or refactor `StateSpaceTimeSeries` model context usage

---

## 🎯 Priority Assessment

| Issue | Priority | Impact | Effort | Phase |
|-------|----------|--------|--------|-------|
| API Inconsistency (data types) | 🔴 Critical | High | Medium | 2 |
| Missing `treated_units` | 🔴 Critical | High | Low | 1 |
| Return Type Inconsistency | 🔴 Critical | High | Low | 1 |
| Code Duplication (5x checks) | 🔴 Critical | Medium | Low | 1 |
| Open/Closed Violation | 🔴 Critical | High | Medium | 2 |
| Special Coordinate Requirements | 🟡 Major | Medium | Medium | 2 |
| Non-Standard Model Context | 🟡 Major | Medium | High | 4 |
| No Prior Configuration | 🟡 Major | Medium | Medium | 2 |
| Complex `_data_setter()` | 🟡 Major | Medium | Medium | 2 |
| Extensive Plotting Conditionals | 🟡 Major | Low | Low | 3 |
| Inconsistent Data Handling | 🟡 Major | Medium | Low | 3 |
| State Management Complexity | 🟡 Major | Low | High | 4 |

---

## 📚 Additional Considerations

### Backward Compatibility
- Changes to model APIs will break existing BSTS user code
- Should version as breaking change (e.g., 0.5.0)
- Consider deprecation warnings before removal

### Testing Requirements
- Add integration tests for time-series model interface
- Test that experiment class works with all model types
- Add tests for data format conversions
- Test prior customization system

### Documentation Needs
- Document time-series model requirements clearly
- Provide migration guide if API changes
- Add examples showing both standard and time-series models
- Document the `TimeSeriesPyMCModel` ABC if created

---

## 🤔 Open Questions

1. **State-space requirements**: Can `StateSpaceTimeSeries` use `self` as context, or does pymc-extras require a separate model?

2. **Backward compatibility**: How many users are already using these experimental models? Should we prioritize backward compatibility or clean API?

3. **Time-series ABC**: Should `TimeSeriesPyMCModel` be a separate class hierarchy, or should we make `PyMCModel` more flexible?

4. **Data format**: Is there value in making BSTS models accept xarray, or is numpy + datetime the right approach for time series?

5. **Prior system**: Should time-series models support dimension-specific priors like `dims=["obs_ind", "treated_units"]`?

---

## 📝 Conclusion

The BSTS implementation adds valuable functionality to CausalPy, but the current approach creates maintenance challenges and API inconsistencies. By following the proposed solutions, we can:

1. Maintain the functionality while improving API consistency
2. Reduce code duplication and improve maintainability
3. Make the codebase more extensible for future time-series models
4. Provide a better user experience with consistent interfaces

The experimental warnings currently in place give us breathing room to make breaking changes if needed. We should prioritize Phase 1 quick wins to address the most critical issues, then move to API standardization in Phase 2.
