# Marimo Notebook Pilot Evaluation

## Summary

**Status**: ⚠️ **NOT RECOMMENDED** for CausalPy documentation at this time.

Marimo notebooks work well as a development tool, but the Sphinx integration embeds them as iframes rather than rendering content natively into the documentation. This creates a different user experience compared to Jupyter notebooks which render inline with the documentation.

---

## What We Tested

1. Converted `did_pymc.ipynb` to Marimo format (`did_pymc_marimo.py`)
2. Integrated with Sphinx via `sphinx-marimo` extension
3. Built and viewed the documentation

## Key Finding

**Marimo notebooks embed as iframes, not inline content.**

| Aspect | Jupyter | Marimo |
|--------|---------|--------|
| **Rendering** | Inline with docs | Embedded iframe |
| **User Experience** | Seamless | "App within a page" |
| **Styling** | Inherits doc theme | Separate marimo theme |
| **Navigation** | Standard page scroll | Iframe scroll |

The marimo content appears in a separate windowed container with its own styling, scrolling, and UI - not integrated into the documentation flow.

---

## Technical Notes

### Making Embedded Notebooks Work

The `sphinx-marimo` extension (v0.3.0) defaults to `html-wasm` export, which requires WebAssembly Python (Pyodide). Since PyMC isn't available in Pyodide, this doesn't work for CausalPy notebooks.

**Fix**: Patch sphinx-marimo to use static HTML export:

```python
# In sphinx_marimo/builder.py, change line ~122:
# FROM: ["marimo", "export", "html-wasm", ...]
# TO:   ["marimo", "export", "html", ...]
```

With this patch, the embedded notebook renders correctly with pre-computed outputs.

### Serving Requirements

The exported HTML must be served via HTTP (not `file://`). Use:
```bash
cd docs/_build && python -m http.server 8000
```

---

## Marimo Advantages (Still Valid)

- ✅ Pure Python format (better git diffs)
- ✅ Reproducible execution (no hidden state)
- ✅ Better IDE support (linting, autocomplete)
- ✅ Reactive development experience

## Why Not Proceed

- ❌ Embedded iframe UX differs from current docs experience
- ❌ PyMC not supported in browser WASM (no true interactivity)
- ❌ `sphinx-marimo` requires patching for static export
- ❌ Users see "Load Interactive Notebook" button before content

---

## Recommendation

**Keep using Jupyter notebooks** for documentation. The current myst-nb integration provides a better user experience with content rendered inline.

Consider marimo for:
- Local development/experimentation
- Interactive tutorials (standalone, not in Sphinx docs)
- Future re-evaluation when sphinx-marimo matures

---

## Files to Clean Up

Remove from this branch before merging:
- `docs/source/notebooks/did_pymc_marimo.py`
- `docs/source/notebooks/did_pymc_marimo.html`
- `docs/source/notebooks/did_pymc_marimo_static.html`
- `docs/source/notebooks/did_pymc_marimo_test.md`

Revert changes to:
- `docs/source/conf.py` (remove sphinx_marimo extension)
- `docs/source/notebooks/index.md` (remove marimo test entry)
- `pyproject.toml` (remove marimo dependencies from docs extras)

---

**Evaluation Date**: December 27, 2024
