# Feature Request: High-Dimensional Fixed Effects Performance Optimization

> **Prerequisite:** This feature builds upon the Panel Fixed Effects PR ([#670](https://github.com/pymc-labs/CausalPy/pull/670)), which must be merged first.

## Summary

Optimize the `within` transformation in `PanelRegression` for very large panels with many units and/or time periods, potentially using Numba or other acceleration techniques.

## Priority

**Lower priority** - CausalPy prioritizes interpretability over speed. However, this becomes relevant for users with large administrative datasets.

## Motivation

### Current Situation

The current `within` transformation uses pure Pandas:

```python
def _within_transform(self, data, formula):
    """Apply within transformation (demeaning) for fixed effects."""
    # Group by unit and/or time, subtract means
    data_demeaned = data.groupby(unit_var).transform(lambda x: x - x.mean())
    ...
```

This works well for moderate-sized panels but may become slow for:
- 100,000+ units
- 1,000+ time periods
- High-frequency panel data

### When This Matters

| Panel Size | Current Performance | User Impact |
|------------|---------------------|-------------|
| 1,000 units × 20 periods | Fast (~seconds) | ✅ No issue |
| 10,000 units × 100 periods | Moderate (~minutes) | ⚠️ Noticeable |
| 100,000 units × 500 periods | Slow (~hours) | ❌ Prohibitive |

### User Story

> "I'm analyzing a firm-level panel with 50,000 firms over 200 quarters. The within transformation takes 20 minutes, and MCMC sampling takes hours. I'd like the data preparation step to be faster."

### Key Insight: MCMC is Usually the Bottleneck

For Bayesian models, MCMC sampling is typically the dominant cost:

| Operation | Typical Time (moderate panel) | Typical Time (large panel) |
|-----------|------------------------------|---------------------------|
| Data loading | Seconds | Minutes |
| Within transformation | Seconds | Minutes-Hours |
| MCMC sampling | Minutes-Hours | Hours-Days |
| **Total** | Minutes-Hours | Hours-Days |

Optimizing demeaning provides limited benefit if sampling takes 10x longer. This feature is lower priority for this reason.

## How pyfixest Handles This

### Multiple Backends

pyfixest offers three demeaning backends:

| Backend | Speed | Dependencies | Availability |
|---------|-------|--------------|--------------|
| Numba | Fast | `numba` | Pure Python, JIT compiled |
| Rust | Fastest | Compiled extension | Pre-built wheels |
| JAX | GPU-accelerated | `jax` | For very large datasets |

### The Algorithm: Alternating Projections

For two-way fixed effects (unit + time), pyfixest uses **iterative alternating projections**:

```
1. Initialize: x_demeaned = x
2. Repeat until convergence:
   a. Demean by units: x_demeaned -= unit_means(x_demeaned)
   b. Demean by time: x_demeaned -= time_means(x_demeaned)
3. Check convergence: |x_new - x_old| < tolerance
```

This converges to the unique solution that removes both fixed effects.

### Rust Implementation

pyfixest's fastest implementation is in Rust (`src/demean.rs`):

```rust
// Key function from pyfixest/src/demean.rs
fn subtract_weighted_group_mean(
    x: &mut [f64],
    sample_weights: &[f64],
    group_ids: &[usize],
    group_weights: &[f64],
    group_weighted_sums: &mut [f64],
) {
    group_weighted_sums.fill(0.0);

    // Accumulate weighted sums per group
    x.iter()
        .zip(sample_weights)
        .zip(group_ids)
        .for_each(|((&xi, &wi), &gid)| {
            group_weighted_sums[gid] += wi * xi;
        });

    // Compute group means
    let group_means: Vec<f64> = group_weighted_sums
        .iter()
        .zip(group_weights)
        .map(|(&sum, &weight)| sum / weight)
        .collect();

    // Subtract means from each sample
    x.iter_mut().zip(group_ids).for_each(|(xi, &gid)| {
        *xi -= group_means[gid];
    });
}
```

Key performance features:
- **Parallel column processing** via Rayon
- **Preallocated buffers** for group sums
- **Vectorized operations** with iterator chains

### Python Wrapper

The Rust code is exposed via PyO3:

```python
# From pyfixest/core/demean.py
from ._core_impl import _demean_rs

def demean(x, flist, weights, tol=1e-08, maxiter=100_000):
    return _demean_rs(
        x.astype(np.float64, copy=False),
        flist.astype(np.uint64, copy=False),
        weights.astype(np.float64, copy=False),
        tol,
        maxiter,
    )
```

## Proposed Solution for CausalPy

### Tiered Approach

Given CausalPy's philosophy (interpretability over speed), a simpler approach is recommended:

1. **Default:** Keep current Pandas implementation (works, readable)
2. **Numba backend:** Optional, for users who have Numba installed
3. **No Rust/JAX:** Too complex for the benefit in CausalPy's use cases

### API

```python
# Default (current behavior)
result = cp.PanelRegression(
    data=df,
    formula="y ~ treated + x1",
    unit_fe_variable="unit",
    fe_method="within",
    model=model
)

# With Numba acceleration (optional)
result = cp.PanelRegression(
    data=df,
    formula="y ~ treated + x1",
    unit_fe_variable="unit",
    fe_method="within",
    demeaner_backend="numba",  # New parameter
    model=model
)

# Auto-detect best available backend
result = cp.PanelRegression(
    ...,
    demeaner_backend="auto",  # Use best available
)
```

### Graceful Degradation

```python
NUMBA_AVAILABLE = False
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    pass

if demeaner_backend == "numba":
    if not NUMBA_AVAILABLE:
        warnings.warn("Numba not available, falling back to Pandas")
        demeaner_backend = "pandas"
```

## Implementation Details

### Numba One-Way Demeaning

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def demean_oneway_numba(data: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Fast group demeaning using Numba.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_obs, n_vars)
    groups : np.ndarray
        1D array of group indices (0 to n_groups-1)

    Returns
    -------
    np.ndarray
        Demeaned data
    """
    n_obs, n_vars = data.shape
    n_groups = groups.max() + 1

    # Compute group sums and counts
    group_sums = np.zeros((n_groups, n_vars))
    group_counts = np.zeros(n_groups)

    for i in range(n_obs):
        g = groups[i]
        group_counts[g] += 1
        for j in range(n_vars):
            group_sums[g, j] += data[i, j]

    # Compute group means
    group_means = np.zeros((n_groups, n_vars))
    for g in range(n_groups):
        if group_counts[g] > 0:
            for j in range(n_vars):
                group_means[g, j] = group_sums[g, j] / group_counts[g]

    # Demean (parallel across observations)
    result = np.zeros_like(data)
    for i in prange(n_obs):  # Parallel loop
        g = groups[i]
        for j in range(n_vars):
            result[i, j] = data[i, j] - group_means[g, j]

    return result
```

### Numba Two-Way Demeaning (Iterative)

For two-way fixed effects (unit + time), use iterative alternating projections:

```python
@jit(nopython=True)
def demean_twoway_numba(
    data: np.ndarray,
    unit_groups: np.ndarray,
    time_groups: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Iterative two-way demeaning (alternating projections).

    This algorithm alternates between removing unit means and time means
    until convergence. It's guaranteed to converge to the unique solution.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_obs, n_vars)
    unit_groups : np.ndarray
        1D array of unit indices
    time_groups : np.ndarray
        1D array of time indices
    max_iter : int
        Maximum iterations before giving up
    tol : float
        Convergence tolerance (max absolute change)

    Returns
    -------
    np.ndarray
        Demeaned data
    """
    result = data.copy()

    for iteration in range(max_iter):
        old_result = result.copy()

        # Demean by units
        result = demean_oneway_numba(result, unit_groups)

        # Demean by time
        result = demean_oneway_numba(result, time_groups)

        # Check convergence
        diff = np.abs(result - old_result).max()
        if diff < tol:
            break

    if iteration == max_iter - 1:
        warnings.warn(f"Two-way demeaning did not converge after {max_iter} iterations")

    return result
```

### Integration with PanelRegression

```python
class PanelRegression(BaseExperiment):

    def __init__(
        self,
        ...,
        demeaner_backend: str = "pandas",  # "pandas", "numba", "auto"
    ):
        self.demeaner_backend = self._resolve_backend(demeaner_backend)
        ...

    def _resolve_backend(self, backend: str) -> str:
        """Resolve backend, falling back if necessary."""
        if backend == "auto":
            return "numba" if NUMBA_AVAILABLE else "pandas"
        elif backend == "numba" and not NUMBA_AVAILABLE:
            warnings.warn("Numba not available, using Pandas")
            return "pandas"
        return backend

    def _within_transform(self, data, formula):
        if self.demeaner_backend == "numba":
            return self._within_transform_numba(data, formula)
        else:
            return self._within_transform_pandas(data, formula)

    def _within_transform_numba(self, data, formula):
        """Fast within transformation using Numba."""
        # Convert groups to integer indices
        unit_codes = pd.factorize(data[self.unit_fe_variable])[0]

        if self.time_fe_variable:
            time_codes = pd.factorize(data[self.time_fe_variable])[0]
            return demean_twoway_numba(data.values, unit_codes, time_codes)
        else:
            return demean_oneway_numba(data.values, unit_codes)
```

## Benchmarking Plan

Before implementing, benchmark to quantify actual speedups:

```python
import time
import pandas as pd
import numpy as np

def benchmark_demeaning(n_units, n_periods, n_vars):
    """Benchmark demeaning implementations."""
    n_obs = n_units * n_periods

    # Generate synthetic panel
    data = np.random.randn(n_obs, n_vars)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # Pandas
    start = time.time()
    df = pd.DataFrame(data)
    df["unit"] = units
    df_demeaned = df.groupby("unit").transform(lambda x: x - x.mean())
    pandas_time = time.time() - start

    # Numba (if available)
    if NUMBA_AVAILABLE:
        # Warm-up JIT
        _ = demean_oneway_numba(data[:100], units[:100])

        start = time.time()
        result = demean_oneway_numba(data, units)
        numba_time = time.time() - start
    else:
        numba_time = None

    return {
        "n_obs": n_obs,
        "pandas_time": pandas_time,
        "numba_time": numba_time,
        "speedup": pandas_time / numba_time if numba_time else None,
    }

# Run benchmarks
for n_units in [1000, 10000, 50000]:
    for n_periods in [10, 50, 100]:
        result = benchmark_demeaning(n_units, n_periods, 10)
        print(f"{n_units:>6} units × {n_periods:>3} periods: "
              f"Pandas={result['pandas_time']:.2f}s, "
              f"Numba={result['numba_time']:.2f}s, "
              f"Speedup={result['speedup']:.1f}x")
```

Expected results (rough):

| Panel Size | Pandas | Numba | Speedup |
|------------|--------|-------|---------|
| 1K × 10 | 0.01s | 0.001s | 10x |
| 10K × 50 | 1s | 0.1s | 10x |
| 50K × 100 | 30s | 3s | 10x |

## What NOT to Implement

Based on CausalPy's philosophy and use cases:

| Feature | Reason to Skip |
|---------|----------------|
| **Rust backend** | Too complex, maintenance burden, requires compiled extensions |
| **JAX/GPU** | Over-engineering; MCMC is the bottleneck, not demeaning |
| **Cython** | Less user-friendly than Numba |
| **Sparse matrix support** | Edge case, not worth complexity |

### Why Not Rust?

pyfixest uses Rust because:
1. They focus on maximum OLS performance
2. They have maintainers with Rust expertise
3. Their users often run millions of regressions

CausalPy differs:
1. Focus is on Bayesian inference (MCMC dominates runtime)
2. Interpretability matters more than speed
3. Users typically run fewer, deeper analyses

## Blockers & Prerequisites

### Required Before Implementation

1. **PR #670 merged:** `PanelRegression` must be stable
2. **Benchmarking:** Need to measure actual speedups in realistic scenarios
3. **User demand:** Should be driven by actual user needs

### Optional Dependencies

- **Numba:** JIT compilation, easy to install via pip/conda
- **No new compiled extensions:** Keep installation simple

## Effort Estimate

| Component | Complexity |
|-----------|------------|
| Numba one-way demeaning | Medium |
| Numba two-way demeaning (iterative) | Medium |
| Backend selection logic | Low |
| Graceful fallback | Low |
| Benchmarking | Medium |
| Tests | Medium |
| Documentation | Low |
| **Total** | **Medium** (~2-3 days) |

## Differentiation from pyfixest

| Aspect | pyfixest | CausalPy (proposed) |
|--------|----------|---------------------|
| Backends | Python, Numba, Rust, JAX | Pandas (default), Numba (optional) |
| Focus | Maximum performance | Reasonable performance, simplicity |
| GPU | JAX/CuPy support | Not planned |
| Compiled extensions | Rust via PyO3 | None |
| Default | Rust (fastest) | Pandas (simplest) |

## Acceptance Criteria

- [ ] Optional `demeaner_backend` parameter added to `PanelRegression`
- [ ] Numba backend implemented for one-way FE
- [ ] Numba backend implemented for two-way FE (iterative)
- [ ] Graceful fallback when Numba not available
- [ ] Warning issued when falling back
- [ ] Benchmark showing speedup for large panels
- [ ] Documentation noting when to use which backend
- [ ] Unit tests verifying identical results across backends

## Related Issues / PRs

- PR #670: `PanelRegression` (prerequisite)
- pyfixest `demean.py` and `src/demean.rs` for reference

## Labels

`enhancement`, `performance`, `low-priority`
