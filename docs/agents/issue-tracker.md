# Issue tracker: `feature_list.json`

Issues and specs for this repo live as entries in `feature_list.json` at the repo root. There is no external tracker: this file is the single source of truth for what work exists, what it depends on, and whether it is done. `CLAUDE.md` requires reading it at session start and updating it at session end.

## Schema

```json
{
  "features": [
    {
      "id": "feat-007",
      "name": "Short imperative title",
      "description": "What to change and the concrete Done condition (command + expected output).",
      "dependencies": ["feat-002"],
      "status": "ready-for-agent",
      "evidence": ""
    }
  ]
}
```

- `id`: `feat-NNN`, zero-padded, next free number. Never reuse or renumber.
- `dependencies`: ids that must be `done` first. A feature is blocked while any of them is not `done`.
- `status`: one of the values in `docs/agents/triage-labels.md`. Exactly one per feature.
- `evidence`: dated lines (`YYYY-MM-DD: ...`). Required when marking `done` (command + observed result) or `wontfix` (reason). Also holds open questions while `needs-info`.

## Conventions

- **Create an issue**: append a new object to `features` with `status: "needs-triage"` and empty `evidence`. If the author writes it with a Done condition, it may start as `ready-for-agent`. Edit with `uv run python` and the `json` module (`indent=2`) so the file stays valid; do not hand-edit with `sed`.
- **Read an issue**: load the file and select by `id`. Read `dependencies` and each dependency's `status` too.
- **List issues**: iterate `features`, filter by `status`. Show `id`, `name`, `status`, `dependencies`.
- **Comment on an issue**: append a dated line to `evidence`. There is no separate comments field.
- **Apply / remove labels**: set `status`. Applying a label replaces the previous one.
- **Close**: set `status` to `done` (with evidence) or `wontfix` (with a reason in `evidence`).

Keep entries small enough to finish in one session. Split anything larger into several features linked by `dependencies`.

## When a skill says "publish to the issue tracker"

Append a feature entry as above and report the new `id`.

## When a skill says "fetch the relevant ticket"

Read the entry with that `id` from `feature_list.json`, plus the entries it depends on.

## Wayfinding operations

Used by `/wayfinder`. The **map** is one feature entry; child tickets are further entries that depend on it.

- **Map**: a feature entry whose `name` starts with `[map]`. Its `description` holds Notes / Decisions-so-far / Fog.
- **Child ticket**: a feature entry with the map's `id` in `dependencies`. Prefix `name` with `[research]`, `[prototype]`, `[grilling]`, or `[task]` for the type.
- **Blocking**: `dependencies`. A ticket is unblocked when every id in `dependencies` (other than the map) has `status: "done"`.
- **Frontier query**: children of the map that are `ready-for-agent` with no open blocker, in file order.
- **Claim**: set the ticket's `status` to `in-progress`. Record it in `progress.md` as the active feature.
- **Resolve**: set `status: "done"`, write the answer into `evidence`, and append a one-line pointer to the map's `description` under Decisions-so-far.
