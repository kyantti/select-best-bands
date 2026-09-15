# Triage Labels

The skills speak in terms of five canonical triage roles. This repo tracks work in `feature_list.json`, which has one `status` field per feature instead of labels. Each role is a `status` value with the same name, and the harness lifecycle (`in-progress`, `done`) uses the same field. Every feature has exactly one `status`, so applying a label replaces the previous one.

## Status values

| Label in mattpocock/skills | `status` in `feature_list.json` | Meaning                                                            |
| -------------------------- | ------------------------------- | ------------------------------------------------------------------ |
| `needs-triage`             | `needs-triage`                  | New. Maintainer has not evaluated it yet.                          |
| `needs-info`               | `needs-info`                    | Waiting on the author for a decision or detail. Question in `evidence`. |
| `ready-for-agent`          | `ready-for-agent`               | Fully specified with a Done condition. The pool agents pick from.  |
| `ready-for-human`          | `ready-for-human`               | Requires the human (real experiment launch, thesis decision, GPU access). |
| _(harness)_                | `in-progress`                   | Being worked on now. At most one feature at a time; named in `progress.md`. |
| _(harness)_                | `done`                          | Finished and verified. `evidence` holds date, command and result.  |
| `wontfix`                  | `wontfix`                       | Closed without action. One-line reason in `evidence`.              |

There is no `not-started` or `blocked` status. "Blocked on another feature" is expressed by `dependencies`; "blocked on the author" is `needs-info`.

## Lifecycle

```
needs-triage ──► ready-for-agent ──► in-progress ──► done
     │                 ▲                  │
     ├──► needs-info ──┘                  └──► needs-info  (agent hit a question; answer in evidence)
     ├──► ready-for-human
     └──► wontfix
```

- `/to-tickets` creates entries as `needs-triage`, unless the author writes the entry with a Done condition, in which case it may start as `ready-for-agent`.
- `/triage` moves `needs-triage` to one of `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
- The startup workflow in `CLAUDE.md` picks one `ready-for-agent` feature whose `dependencies` are all `done`, sets it `in-progress`, and closes it as `done` with evidence.
- An `in-progress` feature that needs the author's decision goes back to `needs-info` with the question appended to `evidence`; it returns to `ready-for-agent` once answered.

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), set `status` to the corresponding value.
