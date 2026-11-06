# API

## Top-level API

```{eval-rst}
.. automodule:: causalpy
   :members:
   :undoc-members:
   :imported-members:
   :exclude-members: EffectSummary, PipelineContext, PipelineResult, SensitivitySummary

.. These dataclasses document their fields in an ``Attributes`` docstring section. With ``:undoc-members:`` autodoc renders every field a second time, so they are documented here with ``:members:`` only.

.. autoclass:: causalpy.EffectSummary
   :members:

.. autoclass:: causalpy.PipelineContext
   :members:

.. autoclass:: causalpy.PipelineResult
   :members:

.. autoclass:: causalpy.SensitivitySummary
   :members:
```

## Modules

```{eval-rst}
.. currentmodule:: causalpy
.. autosummary::
  :recursive:
  :toctree: generated/

  constants
  data
  input_data
  pymc_models
  skl_models
  pymc_forecast_models
  experiments
  pipeline
  reporting
  steps
  checks
```
