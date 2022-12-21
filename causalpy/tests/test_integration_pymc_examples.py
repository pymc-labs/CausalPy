import pandas as pd
import pytest

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


@pytest.mark.integration
def test_did():
    df = cp.load_data("did")
    result = cp.pymc_experiments.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group + t + treated:group",
        time_variable_name="t",
        group_variable_name="group",
        treated=1,
        untreated=0,
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.DifferenceInDifferences)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_did_banks():
    df = (
        cp.load_data("banks")
        .filter(items=["bib6", "bib8", "year"])
        .rename(columns={"bib6": "Sixth District", "bib8": "Eighth District"})
        .groupby("year")
        .median()
    )
    df.reset_index(level=0, inplace=True)
    df_long = pd.melt(
        df,
        id_vars=["year"],
        value_vars=["Sixth District", "Eighth District"],
        var_name="district",
        value_name="bib",
    ).sort_values("year")
    df_long["district"] = df_long["district"].astype("category")
    df_long["unit"] = df_long["district"]
    df_long["treated"] = (df_long.year >= 1931) & (df_long.district == "Sixth District")
    result = cp.pymc_experiments.DifferenceInDifferences(
        df_long[df_long.year.isin([1930, 1931])],
        formula="bib ~ 1 + district + year + district:treated",
        time_variable_name="year",
        group_variable_name="district",
        treated="Sixth District",
        untreated="Eighth District",
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.DifferenceInDifferences)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_rd():
    df = cp.load_data("rd")
    result = cp.pymc_experiments.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + bs(x, df=6) + treated",
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.5,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.RegressionDiscontinuity)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_rd_drinking():
    df = (
        cp.load_data("drinking")
        .rename(columns={"agecell": "age"})
        .assign(treated=lambda df_: df_.age > 21)
    )
    result = cp.pymc_experiments.RegressionDiscontinuity(
        df,
        formula="all ~ 1 + age + treated",
        running_variable_name="age",
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=21,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.RegressionDiscontinuity)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_its():
    df = cp.load_data("its")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.pymc_experiments.SyntheticControl(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_its_covid():
    df = cp.load_data("covid")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    treatment_time = pd.to_datetime("2020-01-01")
    result = cp.pymc_experiments.SyntheticControl(
        df,
        treatment_time,
        formula="standardize(deaths) ~ 0 + standardize(t) + C(month) + standardize(temp)",  # noqa E501
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_sc():
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.pymc_experiments.SyntheticControl(
        df,
        treatment_time,
        formula="actual ~ 0 + a + b + c + d + e + f + g",
        prediction_model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_sc_brexit():
    df = cp.load_data("brexit")
    df["Time"] = pd.to_datetime(df["Time"])
    df.set_index("Time", inplace=True)
    df = df.iloc[df.index > "2009", :]
    treatment_time = pd.to_datetime("2016 June 24")
    df = df.drop(["Japan", "Italy", "US", "Spain"], axis=1)
    target_country = "UK"
    all_countries = df.columns
    other_countries = all_countries.difference({target_country})
    all_countries = list(all_countries)
    other_countries = list(other_countries)
    formula = target_country + " ~ " + "0 + " + " + ".join(other_countries)
    result = cp.pymc_experiments.SyntheticControl(
        df,
        treatment_time,
        formula=formula,
        prediction_model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_ancova():
    df = cp.load_data("anova1")
    result = cp.pymc_experiments.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.PrePostNEGD)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_geolift1():
    df = cp.load_data("geolift1")
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    treatment_time = pd.to_datetime("2022-01-01")
    result = cp.pymc_experiments.SyntheticControl(
        df,
        treatment_time,
        formula="""Denmark ~ 0 + Austria + Belgium + Bulgaria + Croatia + Cyprus
        + Czech_Republic""",
        prediction_model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.pymc_experiments.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
