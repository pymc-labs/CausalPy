# Posting Review Comments

How to draft, present, and post review comments. Two non-negotiable rules:

1. **Never post without explicit human approval.** Always draft first, present in chat, wait.
2. **Cite specific code locations.** Use `line:line:filepath` references when calling out code, not vague paraphrases.

## Comment shapes

### Issue-thread comment (general PR conversation)

For multi-item review summaries, scope/process feedback, or replies to other comments. Posted via `gh pr comment`.

```bash
gh pr comment <num> --body "$(cat <<'EOF'
<text>
EOF
)"
```

### Inline review comment (line-anchored)

For specific code-level suggestions tied to a line range. Posted via `gh api repos/:owner/:repo/pulls/<num>/comments` with JSON body.

Use inline comments for:

- Suggestions that point at a specific line.
- Bug call-outs where the location is the whole point.

Use issue-thread comments for:

- Multi-section review summaries.
- Replies to other comments.
- Process / scope feedback.
- Status updates ("rebase looks clean", "verified all feedback addressed").

### Formal review (`gh pr review`)

For `APPROVE`, `REQUEST_CHANGES`, or `COMMENT` events. **Never** use `--approve` or `--request-changes` without explicit user instruction. `--comment` is fine for grouping multiple inline comments under a single review.

## Templates

### Multi-item review summary

```markdown
Thanks for the PR! [One sentence positive framing if warranted.]

I have comments grouped by severity below.

---

## Must-fix

### 1. [Concise title] (`path/to/file.py`)

[2–4 sentence description. Include why it's a problem, not just what.]

### 2. [Concise title] (`path/to/file.py`)

[Description]

---

## Should-fix

### 3. [Concise title]

[Description, optionally with a code reference:]

` ` `123:128:path/to/file.py
def foo():
    return self.x
` ` `

---

## Nits

### 4. [Concise title]

[Description]

---

## Things I liked

- [Specific positive — generic praise is unhelpful]
- [Another specific positive]
```

The "Things I liked" section is not optional. Reviews that only criticise lose signal and demoralise. Find at least 2 specific things to call out, even on PRs with serious issues.

### Status update on a contributor's response

```markdown
Thanks @<contributor> — confirmed [the rebase / the fix / etc.] (PR is now <state>, [verification details]).

I've gone through round-1 and round-2 review items against the current branch and the code is basically in good shape. Everything is addressed, with tests for the substantive changes ([list 2-3 specifics]). Docs items are also done: [specifics].

One judgement call worth flagging:

- [Item still in question]: [their decision] vs. [your original ask]. [Ask for confirmation that the current shape is intentional.]

[Optional: anything correctly deferred to follow-up]

I'll do a manual pass over [the notebook / the API surface] before approving, but everything is looking good so far.
```

### Inline code-shape suggestion

```markdown
## Suggestion: [concise title]

[Brief context — what does the code do today, and why is the suggestion worth considering?]

### Why it [belongs in / should change]

[Substantive reasoning, with bullet points if there are several supporting facts.]

### Suggested shape

[Concrete proposal, often with two options:]

1. **[Option A]** — [description]
2. **[Option B]** — [description]

I'd lean toward **(1) [or whichever]** because [reason].

### Cleanups before [doing the change]

- [Specific cleanup 1]
- [Specific cleanup 2]

### Scope

[Make explicit whether this is in-scope for the current PR or a follow-up. Defaulting to "happy for follow-up PR" reduces friction unless the change is small.]
```

### Correction

If you posted something incorrect, post a follow-up correction promptly. Don't try to edit silently.

```markdown
Correction to my previous comment: [the wrong claim] was wrong because [what tripped you up]. [The correct statement.] [Anything that does still hold from the previous comment.]
```

## Tone calibration

For each draft, offer the user 2–3 framing variants so they pick:

1. **As-written** — the default tone you drafted.
2. **Stricter** — convert a soft ask to a block, or flip a default to required.
3. **Looser** — accept the contributor's choice with a note, defer the change to a follow-up issue.

Example presentation in chat:

> Three framing options:
> 1. As-written — single soft question, otherwise approval-pending.
> 2. Stricter — request the default flip in this PR (saves a deprecation cycle later).
> 3. Looser — accept everything as-is and approve, file a follow-up issue.

## Attribution

If past comments on the PR have used a reviewer-bot identity (e.g. `claude-opus-4-7-xhigh`), preserve that voice when posting agent-authored comments. The human reviewer's voice and the agent's voice should be distinguishable in the thread.

When in doubt, use a header like:

```markdown
`<model-slug>` here. [Bot disclaimer / scope.]
```

## After posting

Always return the URL to the user:

```bash
# gh pr comment / gh api comment will print the URL on success
```

Echo it back: `Posted: [#826 (comment)](URL).`
