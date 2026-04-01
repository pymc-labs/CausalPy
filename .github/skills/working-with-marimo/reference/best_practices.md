# Marimo Best Practices

## CausalPy Specifics
*   **Data**: Use **Pandas** (standard for CausalPy), even though generic marimo docs suggest Polars.
*   **Plotting**: Use **Matplotlib**, **Seaborn**, or **Arviz**. **Avoid Altair**.
*   **Display**: Return the figure/axis object or use `plt.gca()` as the last expression. **Do NOT use `plt.show()`**.

## Code Structure
*   **Decorators**: Every cell must start with `@app.cell` and define a function `def _():`.
*   **Imports**: Put all imports in one cell (usually the first). Always import `marimo as mo`.
*   **No Globals**: Variables are local to cells unless returned; marimo handles state passing.
*   **Reactivity**: Cells run automatically when inputs change. **Avoid cycles**.

## Visualizations & Outputs
*   **Separation**: Do NOT mix `mo.md` and plots in the same cell. Create a markdown cell, then a plot cell.
*   **Last Expression**: The last line of a cell is automatically displayed.

## Data & SQL
*   **DuckDB**: Use `df = mo.sql(f"""SELECT ...""")`.
*   **Comments**: Do NOT put comments inside `mo.sql` strings or Markdown cells.

## UI Elements
*   **Access**: Use `.value` (e.g., `slider.value`).
*   **Definition**: Define UI element in one cell, access value in another to avoid cycles.

## Example Cell
```python
@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    ax # Display as last expression
    return fig, ax, mo
```
