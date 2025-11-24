# Implementation Plan: Three-Period Interrupted Time Series Design

## Overview

This document outlines the implementation plan for extending the `InterruptedTimeSeries` class to support a three-period design (pre-intervention, intervention, and post-intervention periods). This enables analysis of temporary interventions and measurement of effect persistence and decay.

## Problem Statement

The current ITS implementation assumes interventions are permanent. Many real-world interventions are temporary (marketing campaigns, policy trials, clinical treatments), requiring analysis of:
1. Immediate effects during the intervention
2. Effect persistence after the intervention ends
3. Effect decay patterns

## Design Principles

1. **Backward Compatibility**: Existing code must continue to work unchanged
2. **Opt-in Feature**: Three-period design activated via optional `treatment_end_time` parameter
3. **Minimal Core Changes**: Reuse existing model fitting and forecasting logic
4. **Model Agnostic**: Works with both PyMC (Bayesian) and sklearn (OLS) models
5. **Index Flexibility**: Supports both datetime and numeric indices

## Implementation Architecture

### Phase 1: Core Infrastructure

#### 1.1 Parameter Addition
- **File**: `causalpy/experiments/interrupted_time_series.py`
- **Change**: Add `treatment_end_time` parameter to `__init__` method
  ```python
  def __init__(
      self,
      data: pd.DataFrame,
      treatment_time: Union[int, float, pd.Timestamp],
      formula: str,
      model: Union[PyMCModel, RegressorMixin] | None = None,
      treatment_end_time: Union[int, float, pd.Timestamp] | None = None,  # NEW
      **kwargs: dict,
  )
  ```
- **Validation**: Ensure `treatment_end_time > treatment_time` and within data range
- **Default**: `None` (maintains two-period behavior)

#### 1.2 Input Validation Enhancement
- **File**: `causalpy/experiments/interrupted_time_series.py`
- **Method**: `input_validation()`
- **Changes**:
  - Add `treatment_end_time` parameter
  - Validate index type compatibility (datetime vs numeric)
  - Ensure `treatment_end_time > treatment_time`
  - Ensure `treatment_end_time` is within data range
  - Raise appropriate exceptions with clear error messages

#### 1.3 Period Splitting Method
- **File**: `causalpy/experiments/interrupted_time_series.py`
- **Method**: `_split_post_period()`
- **Purpose**: Split post-intervention data into intervention and post-intervention periods
- **Logic**:
  1. Create boolean masks based on `treatment_end_time`
  2. Split `datapost` into `data_intervention` and `data_post_intervention`
  3. Split `post_pred` into `intervention_pred` and `post_intervention_pred`
     - **Important**: These are slices/views of `post_pred`, not new computations
     - `intervention_pred` = slice of `post_pred` from `treatment_time` to `treatment_end_time`
     - `post_intervention_pred` = slice of `post_pred` from `treatment_end_time` onward
     - No additional model predictions needed—just slicing the existing forecast
  4. Split `post_impact` into `intervention_impact` and `post_intervention_impact`
     - Similarly, these are slices of the existing `post_impact` calculation
  5. Calculate cumulative impacts for each period using the sliced impacts
- **Model Support**:
  - **PyMC**: Use xarray `.sel()` with time dimension (`obs_ind` or `datetime_index`)
  - **OLS**: Use numpy array indexing with position-based selection
- **Time Dimension Handling**: Detect `datetime_index` vs `obs_ind` automatically
- **Key Insight**: The model makes one continuous forecast (`post_pred`), which is then sliced into two periods for analysis. No additional statistical computation is required.

#### 1.4 Integration Point
- **Location**: End of `__init__` method, after `post_impact_cumulative` calculation
- **Logic**:
  ```python
  if self.treatment_end_time is not None:
      self._split_post_period()
  ```

### Phase 2: Reporting Enhancements

#### 2.1 Statistics Computation Enhancement
- **File**: `causalpy/reporting.py`
- **Function**: `_compute_statistics()`
- **Change**: Add `time_dim` parameter (default: `"obs_ind"`)
- **Impact**: Replace hardcoded `"obs_ind"` with parameter to support `datetime_index`
- **Locations**:
  - `impact.mean(dim=time_dim)`
  - `impact.cumsum(dim=time_dim)`
  - `cum_effect.isel({time_dim: -1})`
  - `counterfactual.mean(dim=time_dim)`
  - `counterfactual.cumsum(dim=time_dim).isel({time_dim: -1})`

#### 2.2 Prose Generation Enhancement
- **File**: `causalpy/reporting.py`
- **Functions**: `_generate_prose()` and `_generate_prose_ols()`
- **Change**: Add `prefix` parameter (default: `"Post-period"`)
- **Impact**: Customize period labels in prose output
- **Usage**:
  - `"During intervention"` for intervention period
  - `"Post-intervention"` for post-intervention period
  - `"Post-period"` for default (backward compatible)

#### 2.3 Effect Summary Override
- **File**: `causalpy/experiments/interrupted_time_series.py`
- **Method**: Override `effect_summary()` from `BaseExperiment`
- **New Parameter**: `period: Literal["intervention", "post", "comparison"] | None`
- **Logic**:
  ```python
  if self.treatment_end_time is not None and period is not None:
      # Handle three-period design
      if period == "intervention":
          # Use intervention_impact and intervention_pred
      elif period == "post":
          # Use post_intervention_impact and post_intervention_pred
      elif period == "comparison":
          # Comparative summary with persistence metrics
          # Shows post-intervention effect as percentage of intervention effect
          # Includes posterior probability that effect persisted
          # Currently raises NotImplementedError (can be implemented in Phase 2 or Phase 5)
  else:
      # Default: use base class implementation (backward compatible)
  ```
- **Model Support**: Handle both PyMC and OLS models appropriately

### Phase 3: Documentation

#### 3.1 API Documentation
- **File**: `causalpy/experiments/interrupted_time_series.py`
- **Location**: Class docstring
- **Content**:
  - Document `treatment_end_time` parameter
  - Document new attributes (when `treatment_end_time` is provided)
  - Provide examples for both two-period and three-period designs
  - Explain use cases and benefits

#### 3.2 Glossary Entries
- **File**: `docs/source/knowledgebase/glossary.rst`
- **Entries**:
  - **Intervention period**: Definition of the active treatment period
  - **Effect persistence**: Definition and measurement approach
  - **Effect decay**: Definition and types (exponential, linear, step)

#### 3.3 Example Notebook
- **File**: `docs/source/notebooks/its_three_period_pymc.ipynb`
- **Content**:
  - Introduction to three-period design
  - Marketing campaign example with simulated data
  - Data simulation with decay dynamics
  - Period-specific effect summaries
  - Persistence calculation demonstration
  - Visualization of three periods
- **Structure**: Follow existing notebook patterns (markdown + code cells)

#### 3.4 Existing Notebook Updates
- **File**: `docs/source/notebooks/its_pymc.ipynb`
- **Change**: Add note about `treatment_end_time` parameter
- **Location**: After "Run the analysis" section
- **Format**: MyST note directive with link to three-period notebook

#### 3.5 Notebook Index Update
- **File**: `docs/source/notebooks/index.md`
- **Change**: Add `its_three_period_pymc.ipynb` to Interrupted Time Series section

### Phase 4: Testing

#### 4.1 Test File Creation
- **File**: `causalpy/tests/test_three_period_its.py`
- **Structure**: Comprehensive test suite following pytest patterns

#### 4.2 Test Coverage

**4.2.1 Basic Functionality**
- Test three-period design with PyMC models (datetime index)
- Test three-period design with PyMC models (integer index)
- Test three-period design with sklearn models (datetime index)
- Test three-period design with sklearn models (integer index)

**4.2.2 Backward Compatibility**
- Test that `treatment_end_time=None` maintains two-period behavior
- Test that existing attributes remain unchanged
- Test that existing methods work without modification

**4.2.3 Effect Summary**
- Test `effect_summary(period="intervention")`
- Test `effect_summary(period="post")`
- Test `effect_summary(period=None)` (default behavior)
- Test `effect_summary()` without period parameter
- Test invalid period parameter raises ValueError
- Test comparison period raises NotImplementedError

**4.2.4 Validation**
- Test `treatment_end_time <= treatment_time` raises ValueError
- Test `treatment_end_time` beyond data range raises ValueError
- Test index type mismatches raise BadIndexException

**4.2.5 Edge Cases**
- Test very short post-intervention period
- Test `treatment_end_time` at data boundary
- Test empty intervention or post-intervention periods

**4.2.6 Attributes**
- Test all new attributes exist when `treatment_end_time` is provided
- Test cumulative impact attributes are calculated correctly
- Test data splits are correct (no overlap, complete coverage)

#### 4.3 Test Fixtures
- `datetime_data`: Synthetic datetime-indexed data with three periods
- `integer_data`: Synthetic integer-indexed data with three periods
- Use `sample_kwargs` for fast PyMC sampling in tests

### Phase 5: Future Enhancements (Optional)

#### 5.1 Persistence Analysis Methods
- **Method**: `analyze_persistence()`
- **Returns**: Dictionary with mean effects, persistence ratio, total effects
- **Status**: Deferred (can be added later)

#### 5.2 Decay Model Fitting
- **Method**: `fit_decay_model(decay_type='exponential')`
- **Purpose**: Fit parametric models to post-intervention impacts
- **Status**: Deferred (can be added later)

#### 5.3 Comparison Period Summary
- **Enhancement**: Implement `period='comparison'` in `effect_summary()`
- **Purpose**: Comparative summary with persistence metrics showing:
  - Post-intervention effect as percentage of intervention effect
  - Posterior probability that some effect persisted
  - Comparison of HDI intervals between periods
- **Status**: Currently raises NotImplementedError (can be implemented in Phase 2 or deferred to Phase 5)

#### 5.4 Enhanced Plotting
- **Enhancement**: Update `plot()` methods to visually distinguish three periods
- **Features**:
  - Different colors/styles for intervention vs post-intervention
  - Vertical line at `treatment_end_time`
  - Separate impact panels or annotations
- **Status**: Deferred (current plotting works but could be enhanced)

## Implementation Order

1. ✅ **Phase 1**: Core infrastructure (parameter, validation, splitting)
2. ✅ **Phase 2**: Reporting enhancements (statistics, prose, effect_summary)
3. ✅ **Phase 3**: Documentation (API docs, glossary, notebooks)
4. ✅ **Phase 4**: Testing (comprehensive test suite)
5. ⏳ **Phase 5**: Future enhancements (optional, can be added incrementally)

## Risk Assessment

### Low Risk
- ✅ Backward compatibility maintained (default behavior unchanged)
- ✅ Minimal changes to core model fitting logic
- ✅ Well-tested existing code paths remain untouched
- ✅ Clear separation between two-period and three-period logic

### Medium Risk
- ⚠️ Time dimension handling (datetime_index vs obs_ind) - mitigated by automatic detection
- ⚠️ Model type differences (PyMC vs OLS) - mitigated by explicit branching

### Mitigation Strategies
- Comprehensive test coverage for all model types and index types
- Clear error messages for validation failures
- Extensive documentation with examples
- Incremental implementation with testing at each phase

## Success Criteria

1. ✅ All existing tests pass without modification
2. ✅ New tests cover all three-period functionality
3. ✅ Backward compatibility verified (two-period design unchanged)
4. ✅ Documentation complete (API docs, glossary, example notebook)
5. ✅ Works with both PyMC and sklearn models
6. ✅ Works with both datetime and numeric indices
7. ✅ Clear error messages for invalid inputs
8. ✅ Period-specific effect summaries functional

## Dependencies

- No new external dependencies required
- Uses existing patsy, xarray, pandas, numpy functionality
- Leverages existing reporting infrastructure

## Timeline Estimate

- **Phase 1 (Core Infrastructure)**: 4-6 hours
- **Phase 2 (Reporting Enhancements)**: 3-4 hours
- **Phase 3 (Documentation)**: 3-4 hours
- **Phase 4 (Testing)**: 4-6 hours
- **Total Core Implementation**: ~14-20 hours
- **Phase 5 (Future Enhancements)**: 4-8 hours (optional)

## Notes

- The implementation follows the existing codebase patterns and conventions
- All changes are additive (no breaking changes)
- The design prioritizes clarity and maintainability
- Future enhancements can be added incrementally without breaking existing functionality
