---
name: feature-exploration
description: Explore unfamiliar APIs, libraries, or implementation behavior with minimal reproducible examples and documented findings. Use when implementation details are unclear and can be resolved by reading docs, inspecting code, and running focused experiments.
---

# Feature Exploration

Use this developer skill when a task depends on API behavior that is unclear from memory or partially documented. The goal is to resolve uncertainty before changing production code.

## Workflow

1. State the uncertainty: name the API, behavior, or integration detail that needs proof.
2. Read the closest authoritative docs or source code before experimenting.
3. Build the smallest reproducible example that answers the question.
4. Run it in the project environment when it imports project code or dependencies.
5. Iterate only until the behavior is understood.
6. Record the finding in the final answer or in the durable project docs if it affects future work.
7. Apply the production change using the confirmed behavior.

## Guardrails

- Do not create throwaway files in tracked locations. If a temporary note is needed, use `.scratch/`.
- Do not leave exploratory scripts behind unless they become real tests or documented examples.
- Prefer adding a proper test under `causalpy/tests/` when the discovered behavior is important to preserve.
- Keep experiments narrow; avoid broad refactors while investigating.
- Follow the repository environment rules in `AGENTS.md` for commands that import CausalPy, PyMC, PyTensor, matplotlib, or repo tooling.

## Output

When using this skill, report:

- The uncertainty investigated.
- The evidence gathered.
- The conclusion.
- Any production change or test that now relies on that conclusion.
