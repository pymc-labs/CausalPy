# Documentation animations

This is the source for the interrupted time series animation embedded on the homepage. Commit both the editable sources here and the built player in `docs/source/_static/animations/`.

## Update the animation

Edit `templates/interrupted-time-series.html`, `player.css`, or `player.js`, then rebuild:

```sh
uv run python scripts/animations/build.py
uv run make html
```

Sphinx serves the generated HTML inside an iframe. Normal docs builds do not fit a model or require a JavaScript build tool. The host script, `docs/source/_static/animation-embed.js`, sizes the iframe and starts playback when it enters view. Playback also supports pause, replay, stage selection, scrubbing and reduced-motion preferences.

## Regenerate the fitted data

The committed JSON is a snapshot so presentation changes do not require sampling. To refit the synthetic example:

```sh
uv run python scripts/animations/generate_data.py
uv run python scripts/animations/build.py
```

The generator simulates 72 monthly observations from the explicit `truth` parameters, including a +18-unit intervention after month 48, and fits through CausalPy's public API with default sampling settings. The NumPy seed fixes the observations; posterior summaries will vary slightly across refits. For each revealed post-intervention month, `effect_summary(window=(start, end), alpha=0.06)` exports the running mean and its 94% HDI from the same fit. The interval is computed for the window average, preserving dependence across monthly draws. Inspect the printed R-hat, effective sample sizes and divergences before replacing the committed snapshot. Use `--output PATH` to inspect a new fit separately.

Both prediction and impact intervals currently describe uncertainty in the expected outcome, not noisy posterior-predictive outcomes. The predictive option is tracked in [issue #295](https://github.com/pymc-labs/CausalPy/issues/295). Metadata in the JSON records the model, interval target and probability. When changing interval targets, update the homepage explanation and player labels together.

The scene's dates, axes and explanatory text are tailored to this dataset. If changing the simulation, check those alongside the exported values. Future methods should get their own scene and data; extract shared drawing components once another method makes the common requirements clear.
