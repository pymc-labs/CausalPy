# Contributor pathways and permissions

Contributions are welcome from the community. This document describes how contributors can grow their responsibilities in the CausalPy project and the GitHub permissions that come with each level.

## Current maintainers

<!-- Update this list as the team evolves -->
- [@drbenvincent](https://github.com/drbenvincent)
- [@juanitorduz](https://github.com/juanitorduz)
- [@NathanielF](https://github.com/NathanielF)
- [@lucianopaz](https://github.com/lucianopaz)

## Principles

- **Earned trust, least privilege:** permissions increase gradually as trust is built.
- **Transparency:** criteria and expectations are clear and applied consistently.
- **Sustainability:** pathways should reduce maintainer load over time.
- **Safety by default:** branch protection and CI checks remain in place regardless of role.
- **Respect:** all contributions (code and non-code) are valued.

## Pathway overview

| Level | GitHub permission | Typical scope | Primary focus |
|---|---|---|---|
| Community participant | N/A | Participate via issues, discussions, and PRs from forks | Reporting, ideas, fixes, docs, examples, tests |
| Triager | Triage | Manage issues/PRs (no write) | Labels, reproductions, routing, housekeeping |
| Collaborator | Write | Contribute directly to branches | Regular PRs, reviews, maintenance |
| Maintainer | Maintain | Manage repository operations | Merging, releases, governance, roadmap |

> Note: Titles are descriptive. GitHub permission is the enforceable access level.

## Levels in detail

### 1) Community participant (public access)

**Who this is for**
- Anyone engaging with the project: users, researchers, educators, and prospective contributors.

**What you can do**
- Open issues (bug reports, feature requests, questions).
- Participate in discussions.
- Submit PRs from forks (code, docs, tests, examples).
- Review PRs by leaving comments and suggestions.

**Expectations**
- Follow the [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
- Prefer small, focused PRs.
- Include tests and docs updates when appropriate.
- Be responsive to reviewer feedback.

**Signals you may be ready for elevated access**
- Consistently helpful participation and good judgment.
- High-quality issue reports (clear repro, version info).
- A track record of merged contributions and constructive reviews.

---

### 2) Triager (Triage)

**Who this is for**
- Contributors who help maintain project hygiene by managing issues and PR flow without changing code directly.

**What you can do (typical)**
- Apply and manage labels.
- Ask for reproductions, logs, environment details.
- Close duplicates, redirect questions to Discussions.
- Keep PRs moving by requesting changes, tagging reviewers, and nudging for updates.

**What you cannot do**
- Merge PRs.
- Change repository settings.

**Expectations**
- Use a consistent labeling taxonomy.
- Be neutral and kind; focus on clarity.
- Escalate ambiguous/controversial decisions to maintainers.

**Suggested criteria**
- Demonstrated helpfulness over time (e.g., 4–8 weeks of consistent triage activity).
- Sound judgment on duplicates, scope, and priority.

**Nomination and granting**
- Maintainers can invite directly, or a contributor can request the role by [opening a GitHub issue](https://github.com/pymc-labs/CausalPy/issues/new).
- Access is reviewed periodically; inactivity may result in stepping down.

---

### 3) Collaborator (Write)

**Who this is for**
- Contributors who actively push changes and can be trusted with direct write access.

**What you can do (typical)**
- Push branches to the main repository.
- Help maintain CI, docs, examples.
- Perform routine maintenance tasks (refactors, dependency updates) within agreed scope.

**Expectations**
- Demonstrate good engineering hygiene: tests, docs, changelog discipline (as applicable).
- Respect backwards compatibility and public API stability.
- Participate in code review (both giving and receiving).

**Suggested criteria**
- Sustained contributions (e.g., multiple merged PRs across at least a few weeks/months).
- High-quality reviews that improve code quality and catch issues.
- Familiarity with project standards and tooling.

**Safety mechanisms**
- Branch protection remains enabled (required checks, review requirements).
- Prefer PR-based changes even for collaborators.

---

### 4) Maintainer (Maintain)

**Who this is for**
- People who help run the project: merging, release coordination, and repository management.

**What you can do (typical)**
- Merge PRs.
- Manage labels and milestones.
- Coordinate releases and ensure release notes are accurate.
- Manage project boards (if used).

**Expectations**
- Consistent review and merge quality.
- Ability to mediate disagreements and drive decisions.
- Active stewardship of community norms.

**Suggested criteria**
- Track record of high-impact contributions and reliable collaboration.
- Demonstrated leadership: mentoring, reviews, triage, roadmap contributions.
- Comfortable with responsible disclosure and security processes (if applicable).

**Onboarding**
- Start with a limited scope (e.g., one module or docs/releases) and expand.

---

## Decision process

### How people are invited
- A maintainer opens a short nomination discussion (or uses an internal maintainer thread) referencing:
  - contributions (PRs/issues/reviews)
  - areas of ownership
  - proposed permission level
- After a short review period, maintainers grant access.

### Resolving disagreements
- Decisions are made by informal consensus among maintainers.
- If consensus cannot be reached, a simple majority decides.
- For project-wide decisions (e.g., major API changes, new maintainers), give at least one week for async discussion before finalizing.

### How to step down
- Anyone can request to step down at any time.
- Access can be reduced after long inactivity to minimize risk.

### A note on Admin access
Admin access is reserved for project leads and is not part of the contributor pathway. Admins handle repository settings, secrets, and GitHub Actions configuration.

## Role expectations checklist

- Communicate clearly and respectfully.
- Default to PR-based workflows.
- Keep changes small and reviewable.
- Prefer documenting decisions in issues/PRs.

---

## Appendix: Quick rubric for promotion

Consider promoting when a contributor reliably demonstrates:
- **Quality:** produces correct changes with appropriate tests/docs.
- **Judgment:** scopes work well and respects compatibility.
- **Collaboration:** responds to review, helps others, communicates.
- **Consistency:** shows up over time rather than one-off activity.
