#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Functions that generate data sets used in examples
"""

import numpy as np
import pandas as pd
from scipy.stats import dirichlet, gamma, norm, uniform
from statsmodels.nonparametric.smoothers_lowess import lowess

default_lowess_kwargs: dict[str, float | int] = {"frac": 0.2, "it": 0}
RANDOM_SEED: int = 8927
rng: np.random.Generator = np.random.default_rng(RANDOM_SEED)


def _smoothed_gaussian_random_walk(
    gaussian_random_walk_mu: float,
    gaussian_random_walk_sigma: float,
    N: int,
    lowess_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates Gaussian random walk data and applies LOWESS.

    :param gaussian_random_walk_mu:
        Mean of the random walk
    :param gaussian_random_walk_sigma:
        Standard deviation of the random walk
    :param N:
        Length of the random walk
    :param lowess_kwargs:
        Keyword argument dictionary passed to statsmodels lowess
    """
    x = np.arange(N)
    y = norm(gaussian_random_walk_mu, gaussian_random_walk_sigma).rvs(N).cumsum()
    filtered = lowess(y, x, **lowess_kwargs)
    y = filtered[:, 1]
    return (x, y)


def generate_synthetic_control_data(
    N: int = 100,
    treatment_time: int = 70,
    grw_mu: float = 0.25,
    grw_sigma: float = 1,
    lowess_kwargs: dict = default_lowess_kwargs,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generates data for synthetic control example.

    :param N:
        Number of data points
    :param treatment_time:
        Index where treatment begins in the generated dataframe
    :param grw_mu:
        Mean of Gaussian Random Walk
    :param grw_sigma:
        Standard deviation of Gaussian Random Walk
    :lowess_kwargs:
        Keyword argument dictionary passed to statsmodels lowess

    Example
    --------
    >>> from causalpy.data.simulate_data import generate_synthetic_control_data
    >>> df, weightings_true = generate_synthetic_control_data(treatment_time=70)
    """
    # 1. Generate non-treated variables
    df = pd.DataFrame(
        {
            "a": _smoothed_gaussian_random_walk(grw_mu, grw_sigma, N, lowess_kwargs)[1],
            "b": _smoothed_gaussian_random_walk(grw_mu, grw_sigma, N, lowess_kwargs)[1],
            "c": _smoothed_gaussian_random_walk(grw_mu, grw_sigma, N, lowess_kwargs)[1],
            "d": _smoothed_gaussian_random_walk(grw_mu, grw_sigma, N, lowess_kwargs)[1],
            "e": _smoothed_gaussian_random_walk(grw_mu, grw_sigma, N, lowess_kwargs)[1],
            "f": _smoothed_gaussian_random_walk(grw_mu, grw_sigma, N, lowess_kwargs)[1],
            "g": _smoothed_gaussian_random_walk(grw_mu, grw_sigma, N, lowess_kwargs)[1],
        }
    )

    # 2. Generate counterfactual, based on weighted sum of non-treated variables. This
    # is the counterfactual with NO treatment.
    weightings_true = dirichlet(np.ones(7)).rvs(1)
    df["counterfactual"] = np.dot(df.to_numpy(), weightings_true.T)

    # 3. Generate the causal effect
    causal_effect = gamma(10).pdf(np.arange(0, N, 1) - treatment_time)
    df["causal effect"] = causal_effect * -50

    # 4. Generate the actually observed data, ie the treated with the causal effect
    # applied
    df["actual"] = df["counterfactual"] + df["causal effect"]

    # 5. apply observation noise to all relevant variables
    for var in ["actual", "a", "b", "c", "d", "e", "f", "g"]:
        df[var] += norm(0, 0.25).rvs(N)

    return df, weightings_true


def generate_time_series_data(
    N: int = 100,
    treatment_time: int = 70,
    beta_temp: float = -1,
    beta_linear: float = 0.5,
    beta_intercept: float = 3,
) -> pd.DataFrame:
    """
    Generates interrupted time series example data

    :param N:
        Length of the time series
    :param treatment_time:
        Index of when treatment begins
    :param beta_temp:
        The temperature coefficient
    :param beta_linear:
        The linear coefficient
    :param beta_intercept:
        The intercept

    """
    x = np.arange(0, N, 1)
    df = pd.DataFrame(
        {
            "temperature": np.sin(x * 0.5) + 1,
            "linear": np.linspace(0, 1, N),
            "causal effect": 10 * gamma(10).pdf(np.arange(0, N, 1) - treatment_time),
        }
    )

    df["deaths_counterfactual"] = (
        beta_intercept + beta_temp * df["temperature"] + beta_linear * df["linear"]
    )

    # generate the actually observed data
    # ie the treated with the causal effect applied
    df["deaths_actual"] = df["deaths_counterfactual"] + df["causal effect"]

    # apply observation noise to all relevant variables
    # NOTE: no observation noise on the linear trend component
    for var in ["deaths_actual", "temperature"]:
        df[var] += norm(0, 0.1).rvs(N)

    # add intercept column of ones (for modeling purposes)
    # This is correctly a column of ones, not beta_intercept, as beta_intercept
    # is already incorporated in the data generation above
    df["intercept"] = np.ones(df.shape[0])

    return df


def generate_time_series_data_seasonal(
    treatment_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Generates 10 years of monthly data with seasonality
    """
    dates = pd.date_range(
        start=pd.to_datetime("2010-01-01"), end=pd.to_datetime("2020-01-01"), freq="ME"
    )
    df = pd.DataFrame(data={"date": dates})
    df = df.assign(
        year=lambda x: x["date"].dt.year,
        month=lambda x: x["date"].dt.month,
        t=df.index,
    ).set_index("date", drop=True)
    month_effect = np.array([11, 13, 12, 15, 19, 23, 21, 28, 20, 17, 15, 12])
    df["y"] = 0.2 * df["t"] + 2 * month_effect[np.asarray(df.month.values) - 1]

    N = df.shape[0]
    idx = np.arange(N)[df.index > treatment_time]
    df["causal effect"] = 100 * gamma(10).pdf(
        np.array(np.arange(0, N, 1)) - int(np.min(idx))
    )

    df["y"] += df["causal effect"]
    df["y"] += norm(0, 2).rvs(N)

    # add intercept
    df["intercept"] = np.ones(df.shape[0])
    return df


def generate_time_series_data_simple(
    treatment_time: pd.Timestamp, slope: float = 0.0
) -> pd.DataFrame:
    """Generate simple interrupted time series data, with no seasonality or temporal
    structure.
    """
    dates = pd.date_range(
        start=pd.to_datetime("2010-01-01"), end=pd.to_datetime("2020-01-01"), freq="ME"
    )
    df = pd.DataFrame(data={"date": dates})
    df = df.assign(
        linear_trend=df.index,
    ).set_index("date", drop=True)
    df["timeseries"] = slope * df["linear_trend"]
    N = df.shape[0]
    df["causal effect"] = (df.index > treatment_time) * 2
    df["timeseries"] += df["causal effect"]
    # add intercept
    df["intercept"] = np.ones(df.shape[0])
    # add observation noise
    df["timeseries"] += norm(0, 0.25).rvs(N)
    return df


def generate_did() -> pd.DataFrame:
    """
    Generate Difference in Differences data

    Example
    --------
    >>> from causalpy.data.simulate_data import generate_did
    >>> df = generate_did()
    """
    # true parameters
    control_intercept = 1
    treat_intercept_delta = 0.25
    trend = 1
    Δ = 0.5
    intervention_time = 0.5

    # local functions
    def outcome(
        t: np.ndarray,
        control_intercept: float,
        treat_intercept_delta: float,
        trend: float,
        Δ: float,
        group: np.ndarray,
        post_treatment: np.ndarray,
    ) -> np.ndarray:
        """Compute the outcome of each unit"""
        return (
            control_intercept
            + (treat_intercept_delta * group)
            + (t * trend)
            + (Δ * post_treatment * group)
        )

    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1] * 10,
            "t": [0.0, 1.0, 0.0, 1.0] * 10,
            "unit": np.concatenate([[i] * 2 for i in range(20)]),
        }
    )

    df["post_treatment"] = df["t"] > intervention_time

    df["y"] = outcome(
        np.asarray(df["t"]),
        control_intercept,
        treat_intercept_delta,
        trend,
        Δ,
        np.asarray(df["group"]),
        np.asarray(df["post_treatment"]),
    )
    df["y"] += rng.normal(0, 0.1, df.shape[0])
    return df


def generate_regression_discontinuity_data(
    N: int = 100, true_causal_impact: float = 0.5, true_treatment_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Generate regression discontinuity example data

    Example
    --------
    >>> import pathlib
    >>> from causalpy.data.simulate_data import generate_regression_discontinuity_data
    >>> df = generate_regression_discontinuity_data(true_treatment_threshold=0.5)
    >>> df.to_csv(
    ...     pathlib.Path.cwd() / "regression_discontinuity.csv", index=False
    ... )  # doctest: +SKIP
    """

    def is_treated(x: np.ndarray) -> np.ndarray:
        """Check if x was treated"""
        return np.greater_equal(x, true_treatment_threshold)

    def impact(x: np.ndarray) -> np.ndarray:
        """Assign true_causal_impact to all treated entries"""
        y = np.zeros(len(x))
        y[is_treated(x)] = true_causal_impact
        return y

    x = np.sort((uniform.rvs(size=N) - 0.5) * 2)
    y = np.sin(x * 3) + impact(x) + norm.rvs(scale=0.1, size=N)

    return pd.DataFrame({"x": x, "y": y, "treated": is_treated(x)})


def generate_ancova_data(
    N: int = 200,
    pre_treatment_means: np.ndarray | None = None,
    treatment_effect: int = 2,
    sigma: int = 1,
) -> pd.DataFrame:
    """
    Generate ANCOVA example data

    Example
    --------
    >>> import pathlib
    >>> from causalpy.data.simulate_data import generate_ancova_data
    >>> df = generate_ancova_data(
    ...     N=200, pre_treatment_means=np.array([10, 12]), treatment_effect=2, sigma=1
    ... )
    >>> df.to_csv(pathlib.Path.cwd() / "ancova_data.csv", index=False)  # doctest: +SKIP
    """
    if pre_treatment_means is None:
        pre_treatment_means = np.array([10, 12])
    group = np.random.choice(2, size=N)
    pre = np.random.normal(loc=pre_treatment_means[group])
    post = pre + treatment_effect * group + np.random.normal(size=N) * sigma
    df = pd.DataFrame({"group": group, "pre": pre, "post": post})
    return df


def generate_geolift_data() -> pd.DataFrame:
    """Generate synthetic data for a geolift example. This will consists of 6 untreated
    countries. The treated unit `Denmark` is a weighted combination of the untreated
    units. We additionally specify a treatment effect which takes effect after the
    `treatment_time`. The timeseries data is observed at weekly resolution and has
    annual seasonality, with this seasonality being a drawn from a Gaussian Process with
    a periodic kernel."""
    n_years = 4
    treatment_time = pd.to_datetime("2022-01-01")
    causal_impact = 0.2

    time = pd.date_range(start="2019-01-01", periods=52 * n_years, freq="W")

    untreated = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech_Republic",
    ]

    df = (
        pd.DataFrame(
            {
                country: create_series(n_years=n_years, intercept=3)
                for country in untreated
            }
        )
        .assign(time=time)
        .set_index("time")
    )

    # create treated unit as a weighted sum of the untreated units
    weights = np.random.dirichlet(np.ones(len(untreated)), size=1)[0]
    df = df.assign(Denmark=np.dot(df[untreated].values, weights))

    # add observation noise
    for col in untreated + ["Denmark"]:
        df[col] += np.random.normal(size=len(df), scale=0.1)

    # add treatment effect
    df["Denmark"] += np.where(df.index < treatment_time, 0, causal_impact)

    # ensure we never see any negative sales
    df = df.clip(lower=0)

    return df


def generate_multicell_geolift_data() -> pd.DataFrame:
    """Generate synthetic data for a geolift example. This will consists of 6 untreated
    countries. The treated unit `Denmark` is a weighted combination of the untreated
    units. We additionally specify a treatment effect which takes effect after the
    `treatment_time`. The timeseries data is observed at weekly resolution and has
    annual seasonality, with this seasonality being a drawn from a Gaussian Process with
    a periodic kernel."""
    n_years = 4
    treatment_time = pd.to_datetime("2022-01-01")
    causal_impact = 0.2
    time = pd.date_range(start="2019-01-01", periods=52 * n_years, freq="W")

    untreated = [
        "u1",
        "u2",
        "u3",
        "u4",
        "u5",
        "u6",
        "u7",
        "u8",
        "u9",
        "u10",
        "u11",
        "u12",
    ]

    df = (
        pd.DataFrame(
            {
                country: create_series(n_years=n_years, intercept=3)
                for country in untreated
            }
        )
        .assign(time=time)
        .set_index("time")
    )

    treated = ["t1", "t2", "t3", "t4"]

    for treated_geo in treated:
        # create treated unit as a weighted sum of the untreated units
        weights = np.random.dirichlet(np.ones(len(untreated)), size=1)[0]
        df[treated_geo] = np.dot(df[untreated].values, weights)
        # add treatment effect
        df[treated_geo] += np.where(df.index < treatment_time, 0, causal_impact)

    # add observation noise to all geos
    for col in untreated + treated:
        df[col] += np.random.normal(size=len(df), scale=0.1)

    # ensure we never see any negative sales
    df = df.clip(lower=0)

    return df


def generate_event_study_data(
    n_units: int = 20,
    n_time: int = 20,
    treatment_time: int = 10,
    treated_fraction: float = 0.5,
    event_window: tuple[int, int] = (-5, 5),
    treatment_effects: dict[int, float] | None = None,
    unit_fe_sigma: float = 1.0,
    time_fe_sigma: float = 0.5,
    noise_sigma: float = 0.2,
    predictor_effects: dict[str, float] | None = None,
    ar_phi: float = 0.9,
    ar_scale: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic panel data for event study / dynamic DiD analysis.

    Creates panel data with unit and time fixed effects, where a fraction of units
    receive treatment at a common treatment time. Treatment effects can vary by
    event time (time relative to treatment). Optionally includes time-varying
    predictor variables generated via AR(1) processes.

    Parameters
    ----------
    n_units : int
        Total number of units (treated + control). Default 20.
    n_time : int
        Number of time periods. Default 20.
    treatment_time : int
        Time period when treatment occurs (0-indexed). Default 10.
    treated_fraction : float
        Fraction of units that are treated. Default 0.5.
    event_window : tuple[int, int]
        Range of event times (K_min, K_max) for which treatment effects are defined.
        Default (-5, 5).
    treatment_effects : dict[int, float], optional
        Dictionary mapping event time k to treatment effect beta_k.
        Default creates effects that are 0 for k < 0 (pre-treatment)
        and gradually increase post-treatment.
    unit_fe_sigma : float
        Standard deviation for unit fixed effects. Default 1.0.
    time_fe_sigma : float
        Standard deviation for time fixed effects. Default 0.5.
    noise_sigma : float
        Standard deviation for observation noise. Default 0.2.
    predictor_effects : dict[str, float], optional
        Dictionary mapping predictor names to their true coefficients.
        Each predictor is generated as an AR(1) time series that varies over time
        but is the same for all units at a given time. For example,
        ``{'temperature': 0.3, 'humidity': -0.1}`` creates two predictors.
        Default None (no predictors).
    ar_phi : float
        AR(1) autoregressive coefficient controlling persistence of predictors.
        Values closer to 1 produce smoother, more persistent series.
        Default 0.9.
    ar_scale : float
        Standard deviation of the AR(1) innovation noise for predictors.
        Default 1.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - unit: Unit identifier
        - time: Time period
        - y: Outcome variable
        - treat_time: Treatment time for unit (NaN if never treated)
        - treated: Whether unit is in treated group (0 or 1)
        - <predictor_name>: One column per predictor (if predictor_effects provided)

    Example
    --------
    >>> from causalpy.data.simulate_data import generate_event_study_data
    >>> df = generate_event_study_data(
    ...     n_units=20, n_time=20, treatment_time=10, seed=42
    ... )
    >>> df.shape
    (400, 5)
    >>> df.columns.tolist()
    ['unit', 'time', 'y', 'treat_time', 'treated']

    With predictors:

    >>> df = generate_event_study_data(
    ...     n_units=10,
    ...     n_time=10,
    ...     treatment_time=5,
    ...     seed=42,
    ...     predictor_effects={"temperature": 0.3, "humidity": -0.1},
    ... )
    >>> df.shape
    (100, 7)
    >>> "temperature" in df.columns and "humidity" in df.columns
    True
    """
    if seed is not None:
        np.random.seed(seed)

    # Default treatment effects: zero pre-treatment, gradual increase post-treatment
    if treatment_effects is None:
        treatment_effects = {}
        for k in range(event_window[0], event_window[1] + 1):
            if k < 0:
                treatment_effects[k] = 0.0  # No anticipation
            else:
                # Gradual treatment effect that increases post-treatment
                treatment_effects[k] = 0.5 + 0.1 * k

    # Determine treated units
    n_treated = int(n_units * treated_fraction)
    treated_units = set(range(n_treated))

    # Generate unit fixed effects
    unit_fe = np.random.normal(0, unit_fe_sigma, n_units)

    # Generate time fixed effects
    time_fe = np.random.normal(0, time_fe_sigma, n_time)

    # Generate predictor time series (if any)
    # Each predictor is an AR(1) series that varies over time but is the same
    # for all units at a given time
    predictors: dict[str, np.ndarray] = {}
    if predictor_effects is not None:
        for predictor_name in predictor_effects:
            predictors[predictor_name] = generate_ar1_series(
                n=n_time, phi=ar_phi, scale=ar_scale
            )

    # Build panel data
    data = []
    for unit in range(n_units):
        is_treated = unit in treated_units
        unit_treat_time = treatment_time if is_treated else np.nan

        for t in range(n_time):
            # Base outcome: unit FE + time FE + noise
            y = unit_fe[unit] + time_fe[t] + np.random.normal(0, noise_sigma)

            # Add predictor contributions to outcome
            if predictor_effects is not None:
                for predictor_name, coef in predictor_effects.items():
                    y += coef * predictors[predictor_name][t]

            # Add treatment effect for treated units in event window
            if is_treated:
                event_time = t - treatment_time
                if (
                    event_window[0] <= event_time <= event_window[1]
                    and event_time in treatment_effects
                ):
                    y += treatment_effects[event_time]

            row = {
                "unit": unit,
                "time": t,
                "y": y,
                "treat_time": unit_treat_time,
                "treated": 1 if is_treated else 0,
            }
            # Add predictor values to the row
            for predictor_name, series in predictors.items():
                row[predictor_name] = series[t]

            data.append(row)

    return pd.DataFrame(data)


# -----------------
# UTILITY FUNCTIONS
# -----------------


def generate_ar1_series(
    n: int,
    phi: float = 0.9,
    scale: float = 1.0,
    initial: float = 0.0,
) -> np.ndarray:
    """
    Generate an AR(1) autoregressive time series.

    The AR(1) process is defined as:
        x_{t+1} = phi * x_t + eta_t, where eta_t ~ N(0, scale^2)

    Parameters
    ----------
    n : int
        Length of the time series to generate.
    phi : float
        Autoregressive coefficient controlling persistence. Values closer to 1
        produce smoother, more persistent series. Must be in (-1, 1) for
        stationarity. Default 0.9.
    scale : float
        Standard deviation of the innovation noise. Default 1.0.
    initial : float
        Initial value of the series. Default 0.0.

    Returns
    -------
    np.ndarray
        Array of length n containing the AR(1) time series.

    Example
    -------
    >>> from causalpy.data.simulate_data import generate_ar1_series
    >>> np.random.seed(42)
    >>> series = generate_ar1_series(n=10, phi=0.9, scale=0.5)
    >>> len(series)
    10
    """
    series = np.zeros(n)
    series[0] = initial
    innovations = np.random.normal(0, scale, n - 1)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + innovations[t - 1]
    return series


def generate_seasonality(
    n: int = 12, amplitude: int = 1, length_scale: float = 0.5
) -> np.ndarray:
    """Generate monthly seasonality by sampling from a Gaussian process with a
    Gaussian kernel, using numpy code"""
    # Generate the covariance matrix
    x = np.linspace(0, 1, n)
    x1, x2 = np.meshgrid(x, x)
    cov = periodic_kernel(
        x1, x2, period=1, length_scale=length_scale, amplitude=amplitude
    )
    # Generate the seasonality
    seasonality = np.random.multivariate_normal(np.zeros(n), cov)
    return seasonality


def periodic_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    period: int = 1,
    length_scale: float = 1.0,
    amplitude: int = 1,
) -> np.ndarray:
    """Generate a periodic kernel for gaussian process"""
    return amplitude**2 * np.exp(
        -2 * np.sin(np.pi * np.abs(x1 - x2) / period) ** 2 / length_scale**2
    )


def create_series(
    n: int = 52,
    amplitude: int = 1,
    length_scale: int = 2,
    n_years: int = 4,
    intercept: int = 3,
) -> np.ndarray:
    """
    Returns numpy tile with generated seasonality data repeated over
    multiple years
    """
    return np.tile(
        generate_seasonality(n=n, amplitude=amplitude, length_scale=2) + intercept,
        n_years,
    )


def generate_staggered_did_data(
    n_units: int = 50,
    n_time_periods: int = 20,
    treatment_cohorts: dict[int, int] | None = None,
    treatment_effects: dict[int, float] | None = None,
    unit_fe_scale: float = 2.0,
    time_fe_scale: float = 1.0,
    sigma: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic panel data with staggered treatment adoption.

    Creates a balanced panel dataset where different cohorts of units receive
    treatment at different times. Supports dynamic treatment effects that vary
    by event-time (time relative to treatment).

    Parameters
    ----------
    n_units : int, default=50
        Total number of units in the panel.
    n_time_periods : int, default=20
        Number of time periods in the panel.
    treatment_cohorts : dict[int, int], optional
        Dictionary mapping treatment time to number of units in that cohort.
        Units not assigned to any cohort are never-treated.
        Default: {5: 10, 10: 10, 15: 10} (3 cohorts of 10 units each,
        leaving 20 never-treated units).
    treatment_effects : dict[int, float], optional
        Dictionary mapping event-time (t - G) to treatment effect.
        Event-time 0 is the first treated period.
        Default: {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5} with constant effect
        of 2.5 for all subsequent periods.
    unit_fe_scale : float, default=2.0
        Scale of unit fixed effects (drawn from Normal(0, unit_fe_scale)).
    time_fe_scale : float, default=1.0
        Scale of time fixed effects (drawn from Normal(0, time_fe_scale)).
    sigma : float, default=0.5
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - unit: Unit identifier
        - time: Time period
        - treated: Binary indicator (1 if treated at time t, 0 otherwise)
        - treatment_time: Time of treatment adoption (np.inf for never-treated)
        - y: Observed outcome
        - y0: Counterfactual outcome (for validation)
        - tau: True treatment effect (for validation)

    Examples
    --------
    >>> from causalpy.data.simulate_data import generate_staggered_did_data
    >>> df = generate_staggered_did_data(n_units=30, n_time_periods=15, seed=42)
    >>> df.head()
       unit  time  treated  treatment_time  ...

    Notes
    -----
    The data generating process is:

    .. math::

        Y_{it} = \\alpha_i + \\lambda_t + \\tau_{it} \\cdot D_{it} + \\varepsilon_{it}

    where :math:`\\alpha_i` is the unit fixed effect, :math:`\\lambda_t` is the
    time fixed effect, :math:`D_{it}` is the treatment indicator, and
    :math:`\\tau_{it}` is the dynamic treatment effect that depends on
    event-time :math:`e = t - G_i`.
    """
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = np.random.default_rng()

    # Default treatment cohorts: 3 cohorts at times 5, 10, 15
    if treatment_cohorts is None:
        treatment_cohorts = {5: 10, 10: 10, 15: 10}

    # Default dynamic treatment effects: ramp up then stabilize
    if treatment_effects is None:
        treatment_effects = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5}

    # Validate cohort assignments don't exceed n_units
    total_treated = sum(treatment_cohorts.values())
    if total_treated > n_units:
        raise ValueError(
            f"Total units in treatment cohorts ({total_treated}) "
            f"exceeds n_units ({n_units})"
        )

    # Generate unit fixed effects
    unit_fe = local_rng.normal(0, unit_fe_scale, n_units)

    # Generate time fixed effects
    time_fe = local_rng.normal(0, time_fe_scale, n_time_periods)

    # Assign treatment times to units
    treatment_times = np.full(n_units, np.inf)  # Default: never treated
    unit_idx = 0
    for g, n_cohort in treatment_cohorts.items():
        treatment_times[unit_idx : unit_idx + n_cohort] = g
        unit_idx += n_cohort

    # Shuffle treatment assignments
    local_rng.shuffle(treatment_times)

    # Build panel data
    rows = []
    for i in range(n_units):
        for t in range(n_time_periods):
            g_i = treatment_times[i]
            is_treated = t >= g_i

            # Counterfactual outcome (no treatment)
            y0 = unit_fe[i] + time_fe[t]

            # Treatment effect based on event-time
            if is_treated:
                event_time = int(t - g_i)
                # Use specified effect or last available effect for later periods
                if event_time in treatment_effects:
                    tau = treatment_effects[event_time]
                else:
                    # Use the effect for the maximum specified event-time
                    max_event_time = max(treatment_effects.keys())
                    tau = treatment_effects[max_event_time]
            else:
                tau = 0.0

            # Add noise
            epsilon = local_rng.normal(0, sigma)

            # Observed outcome
            y = y0 + tau + epsilon

            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "treated": int(is_treated),
                    "treatment_time": g_i,
                    "y": y,
                    "y0": y0,
                    "tau": tau,
                }
            )

    df = pd.DataFrame(rows)
    return df
