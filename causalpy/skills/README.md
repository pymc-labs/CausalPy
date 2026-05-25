# User-Facing Agent Skills

Markdown skills in this directory ship inside the `causalpy` wheel so the installed version always matches the library API. They teach AI coding agents how to use CausalPy for causal inference tasks.

## Layout

| Path | Purpose |
|------|---------|
| `designing-experiments/` | Choosing the right quasi-experimental method |
| `performing-causal-analysis/` | Fitting models, estimating impacts, plotting results |
| `running-placebo-analysis/` | Placebo-in-time sensitivity checks |
| `_cli.py` | `causalpy skills install|uninstall|list|check` CLI |
| `_installer.py` | Skill discovery, platform detection, install/uninstall logic |
| `_platforms/` | Platform adapters (Cursor, Claude, Copilot, Windsurf, generic) |
| `_generate_llms_txt.py` | Builds the hosted `llms.txt` for docs |

## Developer skills

Developer-focused skills (environment setup, PR workflows, testing conventions, etc.) live in `.github/skills/` and are auto-discovered in-repo via platform symlinks. They are **not** packaged in the wheel.
