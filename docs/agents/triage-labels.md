# Triage Labels

The skills speak in terms of five canonical triage roles. This repo records them as the `Status:` line of each issue file under `.scratch/` (see `issue-tracker.md`). The harness lifecycle (`in-progress`, `done`) uses the same line. Every issue has exactly one `Status:`, so applying a label replaces the previous one.

## Status values

| Label in mattpocock/skills | `Status:` in the issue file | Meaning                                                            |
| -------------------------- | --------------------------- | ------------------------------------------------------------------ |
| `needs-triage`             | `needs-triage`              | New. Maintainer has not evaluated it yet.                          |
| `needs-info`               | `needs-info`                | Waiting on the author for a decision or detail. Question under `## Comments`. |
| `ready-for-agent`          | `ready-for-agent`           | Fully specified with a Done-when section. The pool agents pick from. |
| `ready-for-human`          | `ready-for-human`           | Requires the human (real experiment launch, thesis decision, GPU access). |
| _(harness)_                | `in-progress`               | Being worked on now. At most one ticket at a time; named in `progress.md`. |
| _(harness)_                | `done`                      | Finished and verified. `## Evidence` holds date, command and result. |
| `wontfix`                  | `wontfix`                   | Closed without action. One-line reason under `## Evidence`.        |

"Blocked on another ticket" is the `Blocked by:` line, not a status. "Blocked on the author" is `needs-info`.

## Lifecycle

```
needs-triage ──► ready-for-agent ──► in-progress ──► done
     │                 ▲                  │
     ├──► needs-info ──┘                  └──► needs-info  (agent hit a question; ask under ## Comments)
     ├──► ready-for-human
     └──► wontfix
```

- `/to-tickets` creates issues as `needs-triage`, unless the author writes them with a Done-when section, in which case they may start as `ready-for-agent`.
- `/triage` moves `needs-triage` to one of `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
- The startup workflow in `CLAUDE.md` picks one `ready-for-agent` ticket whose `Blocked by:` tickets are all `done`, sets it `in-progress`, and closes it as `done` with evidence.
- An `in-progress` ticket that needs the author's decision goes back to `needs-info` with the question under `## Comments`; it returns to `ready-for-agent` once answered.

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), set the `Status:` line to the corresponding value.
