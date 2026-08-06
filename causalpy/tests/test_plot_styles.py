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
"""Tests for the configurable plot themes in :mod:`causalpy.plot_styles`."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D

import causalpy as cp
import causalpy.plot_styles as ps

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


@pytest.fixture(autouse=True)
def _restore_active_style():
    """Prevent the module-global active style from leaking across tests."""
    previous = ps.active_theme().name
    yield
    ps.set_plot_style(previous)


def _close(a, b, atol=0.01):
    return np.allclose(mcolors.to_rgba(a), mcolors.to_rgba(b), atol=atol)


# -- Registry / default-preservation ----------------------------------------
def test_default_theme_is_arviz_darkgrid_and_unchanged():
    """The default must stay arviz-darkgrid with the original accent colours."""
    theme = ps.active_theme()
    assert theme.name == "arviz-darkgrid"
    # rcparams is None => defer to arviz's style library (historical behaviour)
    assert theme.rcparams is None
    assert theme.title_font is None
    assert theme.colors == ps.PlotColors(impact="green", reference="k")


def test_available_styles_includes_editorial():
    assert "arviz-darkgrid" in ps.available_styles()
    assert "editorial" in ps.available_styles()


def test_get_theme_unknown_raises():
    with pytest.raises(ValueError, match="Unknown plot style"):
        ps.get_theme("does-not-exist")


def test_editorial_theme_uses_pymc_brand_palette():
    theme = ps.get_theme("editorial")
    cycle = theme.rcparams["axes.prop_cycle"].by_key()["color"]
    assert cycle[0] == ps.PYMC_TEAL
    assert cycle[1] == ps.PYMC_CHARCOAL
    # ink-role accents are the PyMC charcoal
    assert theme.colors.impact == ps.PYMC_CHARCOAL
    assert theme.title_font[0] == "STIX Two Text"
    assert theme.rcparams["figure.facecolor"] == ps.EDITORIAL_BG


# -- Switching the active style ----------------------------------------------
def test_set_plot_style_switches_and_returns_previous():
    previous = ps.set_plot_style("editorial")
    assert previous == "arviz-darkgrid"
    assert ps.active_theme().name == "editorial"
    assert ps.active_colors().impact == ps.PYMC_CHARCOAL


def test_set_plot_style_unknown_raises_without_changing_state():
    with pytest.raises(ValueError):
        ps.set_plot_style("nope")
    assert ps.active_theme().name == "arviz-darkgrid"


def test_plot_style_context_manager_activates_and_restores():
    assert ps.active_theme().name == "arviz-darkgrid"
    with ps.plot_style("editorial") as theme:
        assert theme.name == "editorial"
        assert ps.active_theme().name == "editorial"
    assert ps.active_theme().name == "arviz-darkgrid"


def test_plot_style_context_restores_on_exception():
    with pytest.raises(RuntimeError), ps.plot_style("editorial"):
        assert ps.active_theme().name == "editorial"
        raise RuntimeError("boom")
    assert ps.active_theme().name == "arviz-darkgrid"


# -- style_context (what plt.style.context receives) -------------------------
def test_style_context_default_defers_to_arviz_library():
    spec = ps.style_context(ps.get_theme("arviz-darkgrid"))
    import arviz as az

    assert spec == [az.style.library["arviz-darkgrid"]]


def test_style_context_editorial_composes_on_matplotlib_default():
    spec = ps.style_context(ps.get_theme("editorial"))
    assert spec[0] == "default"
    assert isinstance(spec[1], dict)
    assert "axes.prop_cycle" in spec[1]
    with plt.style.context(spec):
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        assert cycle[0] == ps.PYMC_TEAL


# -- apply_title_font --------------------------------------------------------
def test_apply_title_font_sets_family():
    fig, ax = plt.subplots()
    ax.set_title("hello")
    ps.apply_title_font(fig, ("STIX Two Text", "DejaVu Serif"))
    assert "STIX Two Text" in ax.title.get_fontfamily()
    plt.close(fig)


def test_apply_title_font_sets_suptitle():
    fig = plt.figure()
    fig.suptitle("super")
    ps.apply_title_font(fig, ("STIX Two Text", "DejaVu Serif"))
    assert "STIX Two Text" in fig._suptitle.get_fontfamily()
    plt.close(fig)


def test_apply_title_font_skips_empty_titles():
    fig, ax = plt.subplots()  # no title set
    ps.apply_title_font(fig, ("DejaVu Serif",))  # must not raise
    plt.close(fig)


# -- bake_cycle_colors -------------------------------------------------------
def test_is_cycle_ref():
    assert ps._is_cycle_ref("C0")
    assert ps._is_cycle_ref("C12")
    assert not ps._is_cycle_ref("#123456")
    assert not ps._is_cycle_ref("red")
    assert not ps._is_cycle_ref((0.1, 0.2, 0.3, 1.0))


def test_bake_cycle_colors_freezes_line_colors():
    with plt.style.context(ps.style_context(ps.get_theme("editorial"))):
        fig, ax = plt.subplots()
        (line,) = ax.plot([0, 1], [0, 1], color="C1")
        # lazily stored as the string "C1" until drawn
        assert line.get_color() == "C1"
        ps.bake_cycle_colors(fig)
        baked = line.get_color()
    # frozen to a concrete RGBA that survives leaving the style context
    assert baked != "C1"
    assert _close(baked, ps.PYMC_CHARCOAL)  # C1 in the editorial cycle
    plt.close(fig)


def test_bake_cycle_colors_leaves_explicit_colors_untouched():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], color="#123456")
    ps.bake_cycle_colors(fig)
    assert _close(line.get_color(), "#123456")
    plt.close(fig)


# -- Integration with a real experiment plot() -------------------------------
@pytest.mark.integration
def test_editorial_theme_applies_to_experiment_plot(mock_pymc_sample, did_data):
    """Editorial theme reaches a real figure: near-white ground + baked colours."""
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    with cp.plot_style("editorial"):
        fig, ax = result.plot(show=False)

    # background is the editorial near-white
    assert _close(fig.get_facecolor(), ps.EDITORIAL_BG)
    # no Line2D retains an unresolved "CN" reference (bake ran)
    line_colors = [ln.get_color() for ln in fig.findobj(Line2D)]
    assert not any(ps._is_cycle_ref(c) for c in line_colors)
    # the PyMC brand teal made it onto the figure
    assert any(_close(c, ps.PYMC_TEAL) for c in line_colors if not ps._is_cycle_ref(c))
    # the context manager restored the default afterwards
    assert ps.active_theme().name == "arviz-darkgrid"
    plt.close(fig)


@pytest.mark.integration
def test_default_theme_plot_not_editorial(mock_pymc_sample, did_data):
    """Under the default theme the figure must NOT pick up the editorial ground."""
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    fig, ax = result.plot(show=False)
    assert not _close(fig.get_facecolor(), ps.EDITORIAL_BG)
    plt.close(fig)
