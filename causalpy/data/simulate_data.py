import numpy as np
import pandas as pd
from scipy.stats import dirichlet, gamma, norm, uniform
from statsmodels.nonparametric.smoothers_lowess import lowess

default_lowess_kwargs = {"frac": 0.2, "it": 0}
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


def _smoothed_gaussian_random_walk(
    gaussian_random_walk_mu, gaussian_random_walk_sigma, N, lowess_kwargs
):
    x = np.arange(N)
    y = norm(gaussian_random_walk_mu, gaussian_random_walk_sigma).rvs(N).cumsum()
    filtered = lowess(y, x, **lowess_kwargs)
    y = filtered[:, 1]
    return (x, y)


def generate_synthetic_control_data(
    N=100,
    treatment_time=70,
    grw_mu=0.25,
    grw_sigma=1,
    lowess_kwargs=default_lowess_kwargs,
):
    """
    Example:
    >> import pathlib
    >> df, weightings_true = generate_synthetic_control_data(
                                treatment_time=treatment_time
                            )
    >> df.to_csv(pathlib.Path.cwd() / 'synthetic_control.csv', index=False)
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
    N=100, treatment_time=70, beta_temp=-1, beta_linear=0.5, beta_intercept=3
):
    x = np.arange(0, 100, 1)
    df = pd.DataFrame(
        {
            "temperature": np.sin(x * 0.5) + 1,
            "linear": np.linspace(0, 1, 100),
            "causal effect": 10 * gamma(10).pdf(np.arange(0, 100, 1) - treatment_time),
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

    # add intercept
    df["intercept"] = np.ones(df.shape[0])

    return df


def generate_time_series_data_seasonal(treatment_time):
    dates = pd.date_range(
        start=pd.to_datetime("2010-01-01"), end=pd.to_datetime("2020-01-01"), freq="M"
    )
    df = pd.DataFrame(data={"date": dates})
    df = df.assign(
        year=lambda x: x["date"].dt.year,
        month=lambda x: x["date"].dt.month,
        t=df.index,
    ).set_index("date", drop=True)
    month_effect = np.array([11, 13, 12, 15, 19, 23, 21, 28, 20, 17, 15, 12])
    df["y"] = 0.2 * df["t"] + 2 * month_effect[df.month.values - 1]

    N = df.shape[0]
    idx = np.arange(N)[df.index > treatment_time]
    df["causal effect"] = 100 * gamma(10).pdf(np.arange(0, N, 1) - np.min(idx))

    df["y"] += df["causal effect"]
    df["y"] += norm(0, 2).rvs(N)

    # add intercept
    df["intercept"] = np.ones(df.shape[0])
    return df


def generate_time_series_data_simple(treatment_time, slope=0.0):
    """Generate simple interrupted time series data, with no seasonality or temporal
    structure.
    """
    dates = pd.date_range(
        start=pd.to_datetime("2010-01-01"), end=pd.to_datetime("2020-01-01"), freq="M"
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


def generate_did():
    # true parameters
    control_intercept = 1
    treat_intercept_delta = 0.25
    trend = 1
    Δ = 0.5
    intervention_time = 0.5

    # local functions
    def outcome(t, control_intercept, treat_intercept_delta, trend, Δ, group, treated):
        return (
            control_intercept
            + (treat_intercept_delta * group)
            + (t * trend)
            + (Δ * treated * group)
        )

    def _is_treated(t, intervention_time, group):
        return (t > intervention_time) * group

    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1] * 10,
            "t": [0.0, 1.0, 0.0, 1.0] * 10,
            "unit": np.concatenate([[i] * 2 for i in range(20)]),
        }
    )

    df["treated"] = _is_treated(df["t"], intervention_time, df["group"])

    df["y"] = outcome(
        df["t"],
        control_intercept,
        treat_intercept_delta,
        trend,
        Δ,
        df["group"],
        df["treated"],
    )
    df["y"] += rng.normal(0, 0.1, df.shape[0])
    return df


def generate_regression_discontinuity_data(
    N=100, true_causal_impact=0.5, true_treatment_threshold=0.0
):
    """
    Example use:
    >> import pathlib
    >> df = generate_regression_discontinuity_data(true_treatment_threshold=0.5)
    >> df.to_csv(pathlib.Path.cwd() / 'regression_discontinuity.csv', index=False)
    """

    def is_treated(x):
        return np.greater_equal(x, true_treatment_threshold)

    def impact(x):
        y = np.zeros(len(x))
        y[is_treated(x)] = true_causal_impact
        return y

    x = np.sort((uniform.rvs(size=N) - 0.5) * 2)
    y = np.sin(x * 3) + impact(x) + norm.rvs(scale=0.1, size=N)

    return pd.DataFrame({"x": x, "y": y, "treated": is_treated(x)})


def generate_ancova_data(
    N=200, pre_treatment_means=np.array([10, 12]), treatment_effect=2, sigma=1
):
    group = np.random.choice(2, size=N)
    pre = np.random.normal(loc=pre_treatment_means[group])
    post = pre + treatment_effect * group + np.random.normal(size=N) * sigma
    df = pd.DataFrame({"group": group, "pre": pre, "post": post})
    return df


def generate_geolift_data():
    """Generate synthetic data for a geolift example. This will consists of 6 untreated
    countries. The treated unit `Denmark` is a weighted combination of the untreated
    units. We additionally specify a treatment effect which takes effect after the
    `treatment_time`. The timeseries data is observed at weekly resolution and has
    annual seasonality, with this seasonality being a drawn from a Gaussian Process with
    a periodic kernel."""
    n_years = 4
    treatment_time = pd.to_datetime("2022-01-01")
    causal_impact = 0.2

    def create_series(n=52, amplitude=1, length_scale=2):
        return np.tile(
            generate_seasonality(n=n, amplitude=amplitude, length_scale=2) + 3, n_years
        )

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
        pd.DataFrame({country: create_series() for country in untreated})
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
    return df


# -----------------
# UTILITY FUNCTIONS
# -----------------


def generate_seasonality(n=12, amplitude=1, length_scale=0.5):
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


def periodic_kernel(x1, x2, period=1, length_scale=1, amplitude=1):
    """Generate a periodic kernal for gaussian process"""
    return amplitude**2 * np.exp(
        -2 * np.sin(np.pi * np.abs(x1 - x2) / period) ** 2 / length_scale**2
    )
