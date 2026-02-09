---
name: working-with-marimo
description: Interactive development in marimo notebooks with validation loops. Use for creating/editing marimo notebooks and verifying execution.
---

# Working with Marimo

Follows a **Plan-Execute-Verify** loop to ensure notebook correctness.

## Feedback Loop

1.  **Context & Plan**:
    *   **Sessions**: `mcp_marimo_get_active_notebooks` (Find session IDs).
    *   **Structure**: `mcp_marimo_get_lightweight_cell_map` (See cell IDs/content).
    *   **Data State**: `mcp_marimo_get_tables_and_variables` (Inspect DataFrames/Variables).
    *   **Cell Detail**: `mcp_marimo_get_cell_runtime_data` (Code, errors, local vars).

2.  **Execute**:
    *   Edit the `.py` file directly using `write` or `search_replace`.
    *   **Rule**: Follow [Best Practices](reference/best_practices.md) (e.g., `@app.cell`, no global state).

3.  **Verify (CRITICAL)**:
    *   **Lint**: `mcp_marimo_lint_notebook` (Static analysis).
    *   **Runtime Errors**: `mcp_marimo_get_notebook_errors` (Execution errors).
    *   **Outputs**: `mcp_marimo_get_cell_outputs` (Visuals/Console).

## Common Commands

*   **Start/Sync**: Marimo automatically syncs file changes.
*   **SQL**: Use `mo.sql` for DuckDB queries.
*   **Plots**: Use `plt.gca()` or return figure. **No `plt.show()`**.

## Reference
See [Best Practices](reference/best_practices.md) for code formatting, reactivity rules, and UI element usage.
