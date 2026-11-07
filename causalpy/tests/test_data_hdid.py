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
Test hierarchical difference-in-differences data generation
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from causalpy.data import hdid
from causalpy.data.simulate_data import generate_hdid_data


@pytest.mark.parametrize("include_covariate_effects", [True, False])
@pytest.mark.parametrize("include_region_effects", [True, False])
def test_generate_hdid_data_flags_only_remove_selected_outcome_contributions(
    include_covariate_effects, include_region_effects
):
    """Flags preserve sampled data and remove only the specified outcome terms."""
    full_panel, params = generate_hdid_data(seed=42)
    panel, _ = generate_hdid_data(
        seed=42,
        include_covariate_effects=include_covariate_effects,
        include_region_effects=include_region_effects,
    )
    pd.testing.assert_frame_equal(
        panel.drop(columns="purchase_amount"),
        full_panel.drop(columns="purchase_amount"),
    )

    outcome = params["config"].outcome
    removed = np.zeros(len(panel))
    if not include_covariate_effects:
        removed += (
            outcome.age_weight * (panel.customer_age - outcome.age_center)
            + outcome.tenure_weight * panel.customer_tenure
            + outcome.history_weight
            * (panel.pre_purchase_history - outcome.history_center)
            + outcome.log_store_size_weight * np.log(panel.store_size)
            + outcome.urban_weight * panel.urban
        )
    if not include_region_effects:
        region = params["config"].region
        effects = dict(zip(region.labels, region.effects, strict=True))
        removed += panel.region.map(effects).astype(float).to_numpy()

    # Both outcomes are rounded to cents, so their difference can differ from
    # the unrounded contribution by up to one cent (plus float32 precision).
    np.testing.assert_allclose(
        full_panel.purchase_amount - panel.purchase_amount,
        removed,
        atol=0.011,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("parameter", "value", "predictor"),
    [
        ("intercept", 82.0, None),
        ("post_effect", 2.0, "post_treatment"),
        ("trend_weight", 2.55, "month"),
    ],
)
def test_generate_hdid_data_outcome_controls(parameter, value, predictor):
    """Each numeric control changes only its corresponding outcome contribution."""
    baseline, _ = generate_hdid_data(seed=42)
    panel, _ = generate_hdid_data(seed=42, **{parameter: value})
    pd.testing.assert_frame_equal(
        panel.drop(columns="purchase_amount"),
        baseline.drop(columns="purchase_amount"),
    )
    expected_change = 2.0 if predictor is None else 2.0 * panel[predictor]
    np.testing.assert_allclose(
        panel.purchase_amount - baseline.purchase_amount,
        expected_change,
        atol=0.011,
        rtol=0,
    )


@pytest.mark.parametrize("seed", [42, 656, 20260296])
def test_generate_hdid_data_reproduces_recovery_configuration(seed):
    """The generation function reproduces the previously validated recovery DGP."""
    config = hdid.Config(
        seed=seed,
        n_stores_total=200,
        n_stores_treated=100,
        n_months=4,
        pre_months=2,
        store=hdid.Store(units_min=5, units_max=5, units_parameters={"lam": 5.0}),
        region=hdid.Region(effects=(0.0, 0.0, 0.0, 0.0)),
        outcome=hdid.Outcome(
            intercept=5.0,
            post_effect=1.0,
            trend_weight=0.0,
            age_weight=0.0,
            tenure_weight=0.0,
            history_weight=0.0,
            log_store_size_weight=0.0,
            urban_weight=0.0,
            store_intercept_parameters={"loc": 0.0, "scale": 1.3},
            treatment_slope_parameters={"loc": 0.0, "scale": 1.5},
            noise_parameters={"loc": 0.0, "scale": 2.0},
        ),
    )
    expected = hdid.HDiDSimulator(config).simulate()
    panel, _ = generate_hdid_data(
        seed=seed,
        n_stores_total=200,
        n_stores_treated=100,
        n_months=4,
        pre_months=2,
        units_min=5,
        units_max=5,
        units_lam=5.0,
        sigma_noise=2.0,
        sigma_treatment_slope=1.5,
        intercept=5.0,
        post_effect=1.0,
        trend_weight=0.0,
        include_covariate_effects=False,
        include_region_effects=False,
    )
    pd.testing.assert_frame_equal(panel, expected)


class TestHDiDSimulationParameterClasses:
    def test_store_sample_dispatches_configured_distribution(self):
        """Validate that Store.sample dispatches to the configured RNG distribution."""
        rng = np.random.default_rng(42)
        store = hdid.Store(
            units_distribution="integers",
            units_parameters={"low": 2, "high": 3},
            urban_distribution="binomial",
            urban_parameters={"n": 1, "p": 1.0},
            size_distribution="normal",
            size_parameters={"loc": 50.0, "scale": 0.0},
        )

        units = store.sample(rng=rng, prefix="units", size=4)
        urban = store.sample(rng=rng, prefix="urban", size=4)
        store_size = store.sample(rng=rng, prefix="size", size=4)

        assert units.tolist() == [2, 2, 2, 2]
        assert urban.tolist() == [1, 1, 1, 1]
        np.testing.assert_allclose(store_size, np.full(4, 50.0))

    def test_customer_sample_dispatches_configured_distribution(self):
        """Validate that Customer.sample dispatches to the configured RNG distribution."""
        rng = np.random.default_rng(42)
        customer = hdid.Customer(
            age_distribution="normal",
            age_parameters={"loc": 41.0, "scale": 0.0},
            tenure_distribution="normal",
            tenure_parameters={"loc": 3.5, "scale": 0.0},
        )

        age = customer.sample(rng=rng, prefix="age", size=3)
        tenure = customer.sample(rng=rng, prefix="tenure", size=3)

        np.testing.assert_allclose(age, np.full(3, 41.0))
        np.testing.assert_allclose(tenure, np.full(3, 3.5))

    def test_history_sample_dispatches_configured_distribution(self):
        """Validate that History.sample dispatches to the configured RNG distribution."""
        rng = np.random.default_rng(42)
        history = hdid.History(
            noise_distribution="normal",
            noise_parameters={"loc": 0.75, "scale": 0.0},
        )

        noise = history.sample(rng=rng, prefix="noise", size=5)

        np.testing.assert_allclose(noise, np.full(5, 0.75))

    def test_outcome_sample_dispatches_configured_distributions(self):
        """Validate that Outcome.sample dispatches each configured outcome distribution."""
        rng = np.random.default_rng(42)
        outcome = hdid.Outcome(
            noise_distribution="normal",
            noise_parameters={"loc": 0.0, "scale": 0.0},
            store_intercept_distribution="normal",
            store_intercept_parameters={"loc": 1.25, "scale": 0.0},
            treatment_slope_distribution="normal",
            treatment_slope_parameters={"loc": -0.5, "scale": 0.0},
        )

        noise = outcome.sample(rng=rng, prefix="noise", size=2)
        store_intercept = outcome.sample(rng=rng, prefix="store_intercept", size=2)
        treatment_slope = outcome.sample(rng=rng, prefix="treatment_slope", size=2)

        np.testing.assert_allclose(noise, np.zeros(2))
        np.testing.assert_allclose(store_intercept, np.full(2, 1.25))
        np.testing.assert_allclose(treatment_slope, np.full(2, -0.5))

    @pytest.mark.parametrize(
        ("parameter", "field_name", "new_value"),
        [
            (hdid.Store(), "units_min", 1),
            (hdid.Customer(), "age_min", 1.0),
            (hdid.History(), "base", 1.0),
            (hdid.Outcome(), "intercept", 1.0),
        ],
    )
    def test_parameter_dataclasses_are_frozen(self, parameter, field_name, new_value):
        """Validate that simulator parameter dataclasses are immutable."""
        with pytest.raises(FrozenInstanceError):
            setattr(parameter, field_name, new_value)

    @pytest.mark.parametrize(
        "parameter", [hdid.Store(), hdid.Customer(), hdid.History(), hdid.Outcome()]
    )
    def test_sample_rejects_unknown_prefix(self, parameter):
        """Validate that unknown sample prefixes raise an attribute error."""
        rng = np.random.default_rng(42)

        with pytest.raises(AttributeError, match="missing_distribution"):
            parameter.sample(rng=rng, prefix="missing", size=1)


class TestHDiDSimulator:
    @staticmethod
    def _small_config(**overrides) -> hdid.Config:
        defaults = {
            "seed": 42,
            "n_stores_total": 4,
            "n_stores_treated": 2,
            "n_months": 4,
            "pre_months": 2,
            "store": hdid.Store(
                units_min=2,
                units_max=2,
                units_distribution="integers",
                units_parameters={"low": 2, "high": 3},
                urban_distribution="binomial",
                urban_parameters={"n": 1, "p": 1.0},
                size_distribution="normal",
                size_parameters={"loc": 50.0, "scale": 0.0},
            ),
            "customer": hdid.Customer(
                age_distribution="normal",
                age_parameters={"loc": 40.0, "scale": 0.0},
                tenure_distribution="normal",
                tenure_parameters={"loc": 4.0, "scale": 0.0},
            ),
            "history": hdid.History(
                store_intercept_weight=0.0,
                urban_weight=0.0,
                noise_distribution="normal",
                noise_parameters={"loc": 0.0, "scale": 0.0},
            ),
            "outcome": hdid.Outcome(
                noise_distribution="normal",
                noise_parameters={"loc": 0.0, "scale": 0.0},
                store_intercept_distribution="normal",
                store_intercept_parameters={"loc": 0.0, "scale": 0.0},
                treatment_slope_distribution="normal",
                treatment_slope_parameters={"loc": 0.0, "scale": 0.0},
            ),
            "run_validation": True,
        }
        defaults.update(overrides)
        return hdid.Config(**defaults)  # type: ignore[arg-type]

    def test_generate_hdid_data_wraps_simulator_generation(self):
        """Validate that generate_hdid_data wraps simulator generation."""
        panel, params = generate_hdid_data(
            seed=123,
            n_stores_total=4,
            n_stores_treated=2,
            n_months=4,
            pre_months=2,
            units_min=2,
            units_max=3,
            units_lam=2.0,
            run_validation=True,
        )

        assert isinstance(panel, pd.DataFrame)
        assert isinstance(params["config"], hdid.Config)
        assert isinstance(params["simulator"], hdid.HDiDSimulator)
        assert params["true_att"] == params["config"].outcome.att_effect
        assert np.isfinite(params["empirical_icc"])
        assert params["n_stores"] == 4
        assert params["n_treated_stores"] == 2
        params["simulator"].validate_panel(panel)

        original_config = hdid.Config(
            seed=123,
            n_stores_total=4,
            n_stores_treated=2,
            n_months=4,
            pre_months=2,
            store=hdid.Store(units_min=2, units_max=3, units_parameters={"lam": 2.0}),
        )
        expected = hdid.HDiDSimulator(original_config).simulate()
        pd.testing.assert_frame_equal(panel, expected)

    def test_validate_config_rejects_invalid_time_window(self):
        """Validate that invalid pre/post time windows are rejected."""
        config = self._small_config(pre_months=0)

        with pytest.raises(ValueError, match="pre_months must be >=1"):
            hdid.HDiDSimulator(config=config)

    def test_validate_config_rejects_invalid_store_unit_bounds(self):
        """Validate that invalid store unit bounds are rejected."""
        store = hdid.Store(units_min=3, units_max=2)
        config = self._small_config(store=store)

        with pytest.raises(
            ValueError, match="store.units_min cannot exceed store.units_max"
        ):
            hdid.HDiDSimulator(config=config)

    def test_init_rejects_region_label_effect_mismatch(self):
        """Validate that region labels and effects must align."""
        config = self._small_config(
            region=hdid.Region(labels=("north",), effects=(0.0, 1.0))
        )

        with pytest.raises(ValueError, match="region.labels and region.effects"):
            hdid.HDiDSimulator(config=config)

    def test_simulate_returns_balanced_store_level_treatment_panel(self):
        """Validate that simulation returns a balanced store-level treatment panel."""
        config = self._small_config()
        panel = hdid.HDiDSimulator(config=config).simulate()

        expected_customers = config.n_stores_total * config.store.units_min
        expected_rows = expected_customers * config.n_months
        required_columns = {
            "store_id",
            "customer_id",
            "month",
            "treated",
            "post_treatment",
            "purchase_amount",
            "customer_age",
            "customer_tenure",
            "pre_purchase_history",
            "store_size",
            "region",
            "urban",
        }

        assert panel.shape[0] == expected_rows
        assert required_columns <= set(panel.columns)
        assert panel["customer_id"].nunique() == expected_customers
        assert (
            panel.groupby("customer_id", observed=True)["month"]
            .nunique()
            .eq(config.n_months)
            .all()
        )
        assert panel.groupby("store_id", observed=True)["treated"].nunique().max() == 1
        assert (
            panel["post_treatment"]
            == (panel["month"] > config.pre_months).astype(np.int8)
        ).all()
        assert list(panel["region"].cat.categories) == list(config.region.labels)
        assert (
            panel["customer_age"]
            .between(config.customer.age_min, config.customer.age_max)
            .all()
        )
        assert (
            panel["customer_tenure"]
            .between(config.customer.tenure_min, config.customer.tenure_max)
            .all()
        )

    def test_simulate_recovers_configured_average_treatment_effect_without_noise(self):
        """Validate that noiseless simulation recovers the configured ATT."""
        config = self._small_config()
        panel = hdid.HDiDSimulator(config=config).simulate()
        means = panel.groupby(["treated", "post_treatment"], observed=True)[
            "purchase_amount"
        ].mean()

        did_estimate = (means.loc[1, 1] - means.loc[1, 0]) - (
            means.loc[0, 1] - means.loc[0, 0]
        )

        assert did_estimate == pytest.approx(config.outcome.att_effect, abs=0.01)

    def test_estimate_icc_returns_finite_ratio(self):
        """Validate that realized generalized ICC returns a finite ratio."""
        config = self._small_config(
            outcome=hdid.Outcome(
                noise_parameters={"loc": 0.0, "scale": 2.0},
                store_intercept_parameters={"loc": 0.0, "scale": 1.0},
                treatment_slope_parameters={"loc": 0.0, "scale": 0.5},
            )
        )
        simulator = hdid.HDiDSimulator(config=config)
        simulator.simulate()

        icc = simulator.estimate_icc()

        assert np.isfinite(icc)
        assert 0.0 <= icc <= 1.0

    def test_estimate_icc_requires_simulation(self):
        """Validate that realized ICC requires a generated sample."""
        simulator = hdid.HDiDSimulator(config=self._small_config())

        with pytest.raises(RuntimeError, match=r"Call simulate\(\)"):
            simulator.estimate_icc()

    def test_save_writes_compressed_csv(self, tmp_path):
        """Validate that simulator save writes a compressed CSV file."""
        config = self._small_config()
        simulator = hdid.HDiDSimulator(config=config)
        panel = simulator.simulate()
        out_path = tmp_path / "hierarchical_did_simulated.csv.gz"

        saved_path = simulator.save(panel=panel, path=out_path)
        loaded = pd.read_csv(saved_path)

        assert saved_path == out_path
        assert saved_path.exists()
        assert loaded.shape == panel.shape
