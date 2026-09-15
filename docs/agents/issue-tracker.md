# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`. It is tracked in git and is the single source of truth for what work exists, what blocks it, and whether it is done. `CLAUDE.md` requires reading it at session start and updating it at session end.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings)
- Blocking is a `Blocked by: NN, NN` line right under `Status:` (`none` when unblocked). A ticket is workable when every listed ticket is `done`
- Each issue has `## Description`, `## Done when` (the concrete check: command + expected output), `## Evidence` (dated lines, filled when closing) and `## Comments`
- Comments and conversation history append to the bottom of the file under the `## Comments` heading

Refer to a ticket as `<feature-slug>/<NN>` (e.g. `repo-health/02`). Never renumber or reuse a number; closed tickets stay in place.

Current feature directories:

- `repo-health/`: make the repo runnable and restartable for agents (tickets 01–06).

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/issues/` (creating the directory and `spec.md` if needed) with `Status: needs-triage`, or `Status: ready-for-agent` when the author writes it with a Done-when section.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path, plus every ticket its `Blocked by:` line names. The user will normally pass the path or the issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
