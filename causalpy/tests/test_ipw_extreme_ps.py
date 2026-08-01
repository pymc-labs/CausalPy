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
Tests for IPW plotting with extreme propensity scores.

Regression tests for issue #645: plot_ate() and plot_balance_ecdf() crash
with ValueError when propensity scores include 0.0 or 1.0 due to
unguarded division.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import causalpy as cp
from causalpy.experiments import inverse_propensity_weighting as ipw_experiment

sample_kwargs = {
    "tune": 50,
    "draws": 100,
    "chains": 2,
    "cores": 2,
    "random_seed": 42,
}


@pytest.fixture(scope="module")
def ipw_result(mock_pymc_sample):
    """Create the NHEFS IPW setup used by the affected notebook."""
    df = cp.load_data("nhefs")
    df_standardised = (df - df.mean()) / df.std()
    df_standardised["trt"] = df["trt"]
    df_standardised["outcome"] = df["outcome"]
    formula = (
        "trt ~ 1 + age + race + sex + smokeintensity + smokeyrs + wt71 + active_1 + "
        "active_2 + education_2 + education_3 + education_4 + education_5 + "
        "exercise_1 + exercise_2"
    )
    return cp.InversePropensityWeighting(
        df_standardised,
        formula=formula,
        outcome_variable="outcome",
        weighting_scheme="robust",
        model=cp.pymc_models.PropensityScore(sample_kwargs=sample_kwargs),
    )


@pytest.fixture
def extreme_idata(ipw_result):
    """Create idata with some propensity scores at 0.0 and 1.0."""
    import copy

    idata = copy.deepcopy(ipw_result.idata)
    idata.posterior["p"][:, :, :5] = 0.0
    idata.posterior["p"][:, :, 5:10] = 1.0
    return idata


@pytest.fixture
def notebook_repro_idata(ipw_result):
    """Create 500 posterior draws with endpoint scores for the notebook call."""
    t = ipw_result.t.flatten()
    treated = np.flatnonzero(t == 1)
    controls = np.flatnonzero(t == 0)
    outcome = np.asarray(ipw_result.y).ravel()
    varying_treated = treated[1:][np.argmax(np.abs(outcome[treated[1:]]))]
    varying_control = controls[1:][np.argmax(np.abs(outcome[controls[1:]]))]
    ps = np.full((1, 500, len(t)), 0.5)
    ps[0, :, treated[0]] = 0.0
    ps[0, :, controls[0]] = 1.0
    ps[0, :, varying_treated] = np.linspace(0.25, 0.75, 500)
    ps[0, :, varying_control] = 0.5 + 0.2 * np.sin(np.linspace(0, np.pi, 500))
    posterior = xr.Dataset({"p": (("chain", "draw", "obs_ind"), ps)})
    return xr.DataTree.from_dict({"posterior": posterior})


@pytest.mark.parametrize("container", ["dataset", "dataarray"])
def test_extract_propensity_draws_normalizes_arviz_containers(
    ipw_result, notebook_repro_idata, monkeypatch, container
):
    """Both ArviZ extraction container forms produce the named propensity draws."""
    p = notebook_repro_idata.posterior["p"]
    extracted = p.to_dataset(name="p") if container == "dataset" else p

    def extract(*_args, **kwargs):
        assert kwargs == {"var_names": "p", "combined": True}
        return extracted

    monkeypatch.setattr(ipw_experiment.az, "extract", extract)
    assert ipw_result._extract_propensity_draws(notebook_repro_idata).identical(p)


@pytest.mark.parametrize(
    "extracted",
    [
        xr.Dataset({"other": xr.DataArray([0.5], dims="sample")}),
        xr.DataArray([0.5], dims="sample", name="other"),
    ],
)
def test_extract_propensity_draws_rejects_missing_p(
    ipw_result, notebook_repro_idata, monkeypatch, extracted
):
    """Unexpected ArviZ outputs fail with the same missing-propensity error."""
    monkeypatch.setattr(
        ipw_experiment.az, "extract", lambda *_args, **_kwargs: extracted
    )
    with pytest.raises(KeyError, match="Posterior propensity score variable 'p'"):
        ipw_result._extract_propensity_draws(notebook_repro_idata)


def test_extract_propensity_draws_normalizes_selection_error(
    ipw_result, notebook_repro_idata, monkeypatch
):
    """ArviZ selection failures use the documented missing-propensity error."""

    def extract(*_args, **_kwargs):
        raise KeyError("p")

    monkeypatch.setattr(ipw_experiment.az, "extract", extract)
    with pytest.raises(KeyError, match="Posterior propensity score variable 'p'"):
        ipw_result._extract_propensity_draws(notebook_repro_idata)


def test_nhefs_notebook_repro_has_finite_plot_data(
    ipw_result, notebook_repro_idata, monkeypatch
):
    """Endpoint draws in the full #645 call stay finite and render a distribution."""
    weighted_histograms = []
    original_histogram = np.histogram

    def record_histogram(a, bins=10, range=None, density=None, weights=None):
        histogram = original_histogram(a, bins, range, density, weights)
        if weights is not None:
            weighted_histograms.append((np.asarray(weights), np.asarray(histogram[0])))
        return histogram

    monkeypatch.setattr(np, "histogram", record_histogram)
    plotted_data = {}
    rendered_bin_counts = {}
    original_axes_hist = plt.Axes.hist

    def record_axes_hist(self, x, *args, **kwargs):
        histogram = original_axes_hist(self, x, *args, **kwargs)
        plotted_data[kwargs["label"]] = np.asarray(x)
        rendered_bin_counts[kwargs["label"]] = np.asarray(histogram[0])
        return histogram

    monkeypatch.setattr(plt.Axes, "hist", record_axes_hist)
    monkeypatch.setattr(ipw_result, "weighting_scheme", "raw")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fig, axs = ipw_result.plot_ate(
            idata=notebook_repro_idata,
            method="robust",
            prop_draws=10,
            ate_draws=500,
        )

    try:
        assert any(
            issubclass(warning.category, UserWarning)
            and "Extreme propensity scores detected" in str(warning.message)
            and "Capping values to prevent numerical instability"
            in str(warning.message)
            for warning in caught_warnings
        )
        assert not any(
            issubclass(warning.category, RuntimeWarning) for warning in caught_warnings
        )
        t = ipw_result.t.flatten()
        treated = t == 1
        controls = t == 0
        outcome = np.asarray(ipw_result.y).ravel()
        clipped_ps = np.clip(
            notebook_repro_idata.posterior["p"].values[0], 1e-6, 1 - 1e-6
        )
        p_of_t = np.mean(t)
        expected_y1 = (outcome[treated] * p_of_t / clipped_ps[:, treated]).sum(
            axis=1
        ) / treated.sum()
        expected_y0 = (
            outcome[controls] * (1 - p_of_t) / (1 - clipped_ps[:, controls])
        ).sum(axis=1) / controls.sum()
        expected_ate = expected_y1 - expected_y0

        assert len(weighted_histograms) == 20
        assert all(weights.size > 0 for weights, _ in weighted_histograms)
        assert all(
            np.isfinite(values).all()
            for weights, counts in weighted_histograms
            for values in (weights, counts)
        )
        bins = np.arange(0, 1.005, 0.005)
        expected_weighted_histograms = []
        for ps in clipped_ps[:10]:
            expected_weighted_histograms.extend(
                [
                    original_histogram(
                        ps[controls],
                        bins=bins,
                        weights=(1 - p_of_t) / (1 - ps[controls]),
                    )[0],
                    original_histogram(
                        ps[treated],
                        bins=bins,
                        weights=p_of_t / ps[treated],
                    )[0],
                ]
            )
        for (_, counts), expected_counts in zip(
            weighted_histograms, expected_weighted_histograms, strict=True
        ):
            np.testing.assert_allclose(counts, expected_counts)
        for (_, control_counts), (_, treated_counts) in zip(
            weighted_histograms[::2], weighted_histograms[1::2], strict=True
        ):
            assert control_counts[-1] > 0
            assert treated_counts[0] > 0
        top_left_edges = np.asarray([patch.get_x() for patch in axs[0].patches])
        top_widths = np.asarray([patch.get_width() for patch in axs[0].patches])
        geometry_atol = np.finfo(float).eps
        np.testing.assert_allclose(
            np.unique(top_left_edges), bins[:-1], rtol=0, atol=geometry_atol
        )
        np.testing.assert_allclose(
            top_widths, np.diff(bins)[0], rtol=0, atol=geometry_atol
        )
        assert np.isclose(
            (top_left_edges + top_widths).max(),
            bins[-1],
            rtol=0,
            atol=geometry_atol,
        )
        top_heights = np.asarray([patch.get_height() for patch in axs[0].patches])
        n_bins = len(bins) - 1
        assert top_heights.shape == (10 * 4 * n_bins,)
        top_heights = top_heights.reshape(10, 4, n_bins)
        assert np.all(top_heights[:, 0] >= 0)
        assert np.all(top_heights[:, 1] <= 0)
        assert np.all(top_heights[:, 2] >= 0)
        assert np.all(top_heights[:, 3] <= 0)
        assert np.any(top_heights[:, 2] > 0)
        assert np.any(top_heights[:, 3] < 0)

        assert set(plotted_data) == {"E(Y(1))", "E(Y(0))", "ATE"}
        assert all(values.shape == (500,) for values in plotted_data.values())
        assert all(np.isfinite(values).all() for values in plotted_data.values())
        np.testing.assert_allclose(plotted_data["E(Y(1))"], expected_y1)
        np.testing.assert_allclose(plotted_data["E(Y(0))"], expected_y0)
        np.testing.assert_allclose(plotted_data["ATE"], expected_ate)
        np.testing.assert_allclose(
            plotted_data["ATE"],
            plotted_data["E(Y(1))"] - plotted_data["E(Y(0))"],
        )
        assert all(
            np.count_nonzero(counts) > 1 for counts in rendered_bin_counts.values()
        )
    finally:
        plt.close(fig)


def _axis_fraction(ax, value):
    """Return the share of the axes height spanned by ``value`` data units."""
    to_axes = ax.transAxes.inverted().transform
    origin = to_axes(ax.transData.transform((0, 0)))[1]
    top = to_axes(ax.transData.transform((0, value)))[1]
    return abs(top - origin)


def test_extreme_scores_keep_observation_counts_visible(
    ipw_result, notebook_repro_idata
):
    """Inflated pseudo-population mass must not flatten the propensity counts."""
    with pytest.warns(UserWarning, match="Extreme propensity scores"):
        fig, axs = ipw_result.plot_ate(
            idata=notebook_repro_idata, method="robust", prop_draws=2, ate_draws=2
        )
    try:
        t = ipw_result.t.flatten()
        clipped_ps = np.clip(
            notebook_repro_idata.posterior["p"].values[0, 0], 1e-6, 1 - 1e-6
        )
        bins = np.arange(0, 1.005, 0.005)
        count_peak = max(
            np.histogram(clipped_ps[t == 0], bins=bins)[0].max(),
            np.histogram(clipped_ps[t == 1], bins=bins)[0].max(),
        )
        heights = np.abs([patch.get_height() for patch in axs[0].patches])
        # The clipped endpoints inflate the pseudo population by orders of
        # magnitude over the observation counts, so a linear axis would leave
        # the propensity score distribution invisible.
        assert heights.max() > 100 * count_peak
        assert axs[0].get_yscale() == "symlog"
        assert axs[0].yaxis.get_transform().linthresh == count_peak
        assert _axis_fraction(axs[0], count_peak) > 0.05
    finally:
        plt.close(fig)


def test_well_behaved_weights_stay_on_the_linear_scale(
    ipw_result, notebook_repro_idata
):
    """Weights that do not exceed the counts render inside the linear region."""
    with pytest.warns(UserWarning, match="Extreme propensity scores"):
        fig, axs = ipw_result.plot_ate(
            idata=notebook_repro_idata, method="overlap", prop_draws=2, ate_draws=2
        )
    try:
        linthresh = axs[0].yaxis.get_transform().linthresh
        heights = np.abs([patch.get_height() for patch in axs[0].patches])
        assert heights.max() <= linthresh
    finally:
        plt.close(fig)


class TestPlotAteExtremeScores:
    """plot_ate must not crash when propensity scores hit 0 or 1."""

    @pytest.mark.parametrize("method", ["raw", "robust", "overlap"])
    def test_plot_ate_no_crash(self, ipw_result, extreme_idata, method):
        """Verify plot_ate renders without error for each weighting scheme."""
        fig, axs = ipw_result.plot_ate(
            idata=extreme_idata, method=method, prop_draws=1, ate_draws=5
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotBalanceEcdfExtremeScores:
    """plot_balance_ecdf must not crash when propensity scores hit 0 or 1."""

    @pytest.mark.parametrize("scheme", ["raw", "robust", "overlap"])
    def test_plot_balance_ecdf_no_crash(self, ipw_result, extreme_idata, scheme):
        """Verify plot_balance_ecdf renders without error for each weighting scheme."""
        fig, axs = ipw_result.plot_balance_ecdf(
            "age", idata=extreme_idata, weighting_scheme=scheme
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.parametrize("idata_fixture", ["extreme_idata", "notebook_repro_idata"])
def test_plot_balance_ecdf_extracts_p_from_dataset_or_dataarray(
    request, ipw_result, idata_fixture
):
    """Balance plotting accepts ArviZ's Dataset and single-DataArray extraction forms."""
    idata = request.getfixturevalue(idata_fixture)
    with pytest.warns(UserWarning, match="Extreme propensity scores"):
        fig, axs = ipw_result.plot_balance_ecdf(
            "age", idata=idata, weighting_scheme="robust"
        )
    try:
        assert sum(len(ax.lines) for ax in axs) == 4
        assert all(
            np.isfinite(line.get_ydata()).all() for ax in axs for line in ax.lines
        )
    finally:
        plt.close(fig)


class TestPreparePs:
    """Unit tests for _prepare_ps clipping behavior."""

    def test_clips_zeros(self, ipw_result):
        """Scores at 0.0 are clipped to eps."""
        ps = np.array([0.0, 0.5, 1.0])
        clipped = ipw_result._prepare_ps(ps)
        assert clipped[0] > 0.0
        assert clipped[2] < 1.0
        assert clipped[1] == 0.5

    def test_warns_on_extreme(self, ipw_result):
        """A warning is emitted when extreme scores are detected."""
        ps = np.array([0.0, 0.5, 1.0])
        with pytest.warns(UserWarning, match="Extreme propensity scores"):
            ipw_result._prepare_ps(ps)

    def test_no_warn_on_safe(self, ipw_result):
        """No warning when all scores are within bounds."""
        ps = np.array([0.3, 0.5, 0.7])
        # Should not warn
        clipped = ipw_result._prepare_ps(ps)
        np.testing.assert_array_equal(ps, clipped)
