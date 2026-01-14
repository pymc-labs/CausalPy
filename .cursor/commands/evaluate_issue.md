# Evaluate Issue

Analyze a GitHub issue, its discussion history, and the current codebase state to provide actionable insights and advance progress toward resolution.

## Purpose

Many issues languish without resolution due to:
- **Unclear next steps**: Decisions need to be made but haven't been
- **Waiting on dependencies**: Blocked by refactors or other PRs
- **Stale context**: The codebase has evolved since the issue was filed
- **Missing information**: More details are needed to proceed

This command helps by:
1. Understanding the full issue context (description + all comments)
2. Checking relevance against the current `main` branch
3. Proposing concrete next steps or requesting specific information

## Prerequisites Check

Before starting, verify the GitHub CLI is available and authenticated:

1. **Check `gh` is installed:**
   ```bash
   gh --version
   ```
   - If not found, guide the user to install it:
     - macOS: `brew install gh`
     - Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md
     - Windows: `winget install --id GitHub.cli`

2. **Check authentication:**
   ```bash
   gh auth status
   ```
   - If not authenticated, guide: `gh auth login`

3. **Sandbox considerations:**
   - GitHub CLI commands require access to credentials stored outside the workspace
   - **Always use `required_permissions: ["all"]`** for any `gh` commands to bypass sandbox restrictions

## Workflow

### Phase 1: Fetch and Parse the Issue

1. **Get the issue number from the user:**
   - The user should provide the issue number when invoking this command
   - If not provided, ask: "Which issue number would you like me to evaluate?"

2. **Fetch full issue details:**
   ```bash
   gh issue view <issue_number> --json number,title,body,state,createdAt,updatedAt,labels,author,comments,assignees
   ```

3. **Parse and summarize:**
   - Extract the core problem or feature request
   - Identify any acceptance criteria mentioned
   - Note files, modules, or APIs referenced
   - Summarize the discussion timeline from comments
   - Identify any blockers, decisions, or open questions raised

### Phase 2: Assess Current Relevance

1. **Check if the issue is still valid:**
   - Search the codebase for files/modules mentioned in the issue
   - Check if the problem described still exists
   - Look for any PRs or commits that may have addressed the issue

2. **Check for related open PRs:**
   ```bash
   gh pr list --search "in:title <keywords>" --json number,title,state,url
   ```

3. **Check for related closed issues or PRs:**
   ```bash
   gh issue list --state closed --search "<keywords>" --json number,title,url --limit 5
   gh pr list --state merged --search "<keywords>" --json number,title,url --limit 5
   ```

4. **Investigate the codebase:**
   - Use semantic search and grep to understand relevant code paths
   - Check if APIs, classes, or methods mentioned still exist and are unchanged
   - Look for tests that might reveal expected behavior

### Phase 3: Analyze and Formulate Recommendation

Based on your findings, determine the appropriate response category:

#### Category A: Issue is resolved or no longer relevant
- The issue has been addressed by merged PRs or commits
- The described problem no longer exists in `main`
- **Action:** Recommend closing with explanation

#### Category B: Blocked by external factors
- Waiting on another PR to merge
- Depends on upstream package changes
- Needs architectural decisions first
- **Action:** Document the blocker and any workarounds

#### Category C: Needs more information
- Problem cannot be reproduced
- Acceptance criteria are unclear
- Missing context about use case or environment
- **Action:** Ask specific, targeted questions

#### Category D: Ready for implementation
- Clear problem with clear solution
- No blockers identified
- Codebase is ready to accept changes
- **Action:** Outline recommended approach or propose options

#### Category E: Needs decision/direction
- Multiple valid approaches exist
- Trade-offs need to be weighed
- Maintainer input is needed
- **Action:** Present options with pros/cons, request decision

### Phase 4: Draft the Comment

1. **Create the `.scratch/issue_comments/` directory if it doesn't exist:**
   ```bash
   mkdir -p .scratch/issue_comments
   ```

2. **Draft the comment in markdown:**

   Create file `.scratch/issue_comments/issue-<number>-evaluation.md` with this structure:

   ```markdown
   *This comment was generated with LLM assistance and may have been edited by the commenter.*

   ---

   ## Issue Evaluation

   ### Summary
   <Brief restatement of the issue and its current status>

   ### Current Relevance
   <Analysis of whether the issue is still valid against current `main`>
   - Code references checked: <list files/modules reviewed>
   - Related PRs: <any open or merged PRs>
   - Changes since issue creation: <relevant commits or changes>

   ### Discussion Summary
   <Key points from the comment thread, if any>

   ### Recommendation
   <Based on your analysis, one of:>
   - **Status:** Ready for implementation / Needs decision / Needs information / Blocked / Can be closed
   - **Proposed next steps:** <specific, actionable items>

   ### [If applicable] Proposed Approach
   <Technical details, options with trade-offs, or specific questions>

   ---

   <Optional: tag relevant maintainers or reference related issues/PRs>
   ```

3. **Present draft to user:**
   > "I've prepared a comment for issue #<number>. Here's the draft:
   >
   > <show full markdown content>
   >
   > Would you like to:
   > 1. **Post as-is** - I'll submit this comment
> 2. **Edit first** - Make changes to the draft file at `.scratch/issue_comments/issue-<number>-evaluation.md`
   > 3. **Revise** - Tell me what to change and I'll update the draft
   > 4. **Cancel** - Don't post anything"

   **Wait for user response.**

### Phase 5: Post the Comment

1. **After user approval, post the comment:**
   ```bash
   gh issue comment <issue_number> --body-file .scratch/issue_comments/issue-<number>-evaluation.md
   ```

2. **Report success:**
   > "✅ Comment posted to issue #<number>!
   >
> The draft has been saved to `.scratch/issue_comments/issue-<number>-evaluation.md` for reference."

3. **Cleanup (optional):**
   - Ask if the user wants to keep or delete the draft file
- The `.scratch/issue_comments/` directory is gitignored, so drafts won't be committed

## Comment Guidelines

### Tone and Style
- Be constructive and helpful
- Acknowledge the original author's work
- Be specific rather than vague
- If recommending closure, be respectful and explain why

### Technical Depth
- Reference specific files, line numbers, or code snippets when relevant
- Link to related issues, PRs, or documentation
- Provide code examples if proposing an approach

### Actionability
- Every comment should either:
  - Propose a clear next step
  - Ask a specific question that will unblock progress
  - Provide a decision with rationale
- Avoid "this looks interesting" without action items

## Error Handling

- **Issue not found:** Verify the issue number and repository
- **Permission denied:** Check `gh auth status` and repository access
- **Issue is locked:** Inform user that comments cannot be added
- **Draft already exists:** Ask if user wants to overwrite or use existing draft

## Important Notes

- **Always wait for user confirmation** before posting comments
- **Be transparent** about your analysis and any limitations
- **Acknowledge uncertainty** - if you're not sure about something, say so
- **The LLM disclaimer is mandatory** - always include the italic note at the top
- **Keep drafts local** - the `.scratch/issue_comments/` directory is gitignored
