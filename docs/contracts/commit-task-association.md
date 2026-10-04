# Commit↔task association: the `Commit` object and the `Commit->Task` link

## Where this lives

The board ontology is the schema-1 normalized document of `tools/board/board.py`
— `space`, `projects`, `milestones`, `tasks`, `writes`, plus the inline `plan` —
documented in `tools/board/README.md`, with the task object normalized by
`normalize_task`. This file is its **commit-side extension**: the `Commit` object
and the link that carries a commit to the task row it serves.

| | |
|---|---|
| gate | `python3 tools/hooks/lane.py self-test` (exit 0; the `lane-ontology` and `lane-capture` controls) |
| projector | `tools/hooks/lane.py` `project_commit`, `commit_links`, `mode_link` |
| carrier | `board: <row-id>` in the commit message, or in the committed change |

## Objects

### `Commit`

Identity: **the commit sha**. A commit object does not exist while `commit-msg`
runs, so a prospective commit is identified by the index digest of the content it
will carry; the durable identity is read back from the committed object by
`lane.py link <rev>`.

| property | meaning |
|---|---|
| sha | git object id of the commit; `""` while the commit is prospective |
| tree | git tree object id of the commit; the sha256 of the git index while prospective |
| author | git author identity, as `git var GIT_AUTHOR_IDENT` reports it for a prospective commit and `git show -s --format=%an <%ae>` for a committed one |
| timestamp | author date, ISO-8601 UTC |
| row | the declared board row's **exact** id, resolved against the enumerated board; `""` in a no-row lane |
| lane | `row`, `housekeeping` or `spike` — which declaration the commit carries |
| paths | the committed paths |
| message_sha256 | sha256 of the commit message the declaration was read from |
| index_sha256 | sha256 of the git index at hook time |
| staged_diff_sha256 | sha256 of `git diff --cached -U0` at hook time |

### `Task`

The board row, normalized by `tools/board/board.py` `normalize_task`: `id`,
`name`, `status_value`, `status_label`, `milestone_id`, `milestone_title`,
`project_id`, `project_name`, `text`, `gates`, `paths`, `receipts`, `plan_ids`,
`internal_by`. Its status vocabulary (`actionable`, `done`, `canceled`, `other`)
is `status_bucket`'s, shared with the driver.

### Link `Commit->Task`

`{"type": "Commit->Task", "source": "commit:<sha|tree>", "target": "task:<row>",
"lane": <lane>}`. Exactly one link per commit in the `row` lane; none in the
`housekeeping` and `spike` lanes, which name no row.

## Projection

The `board:` message field is **one projection** of the link, not the link
itself:

1. a commit writes `board: <row-id>` (or one of the two no-row markers);
2. `find_declaration` reads the field from the commit message and from the
   committed change; two sources naming different rows is
   `board-declaration-conflict`;
3. `match_rows` resolves the token by **exact** row id, or by a plan row id that
   exactly one row's own description declares. Name-slug and row-id-prefix
   matching are not part of the commit path: a fuzzy rule is how a declaration
   for one row resolved to a different, older row;
4. `check_row` requires the resolved row to be an open row — never `Done` or
   `Canceled` (`board-row-closed`);
5. `check_commit` requires the resolved row id to appear in the committed change
   (`board-row-not-in-commit`), so the association is readable from the commit
   and not only from the message that declared it;
6. `project_commit` writes the `Commit` object and the link into the schema-1
   document, and `commit_links` reads the link back out of the document. A link
   that exists only in the writer is not a link.

## Rules the association is checked against

| rule | refusal |
|---|---|
| `board-row-not-found` | the token names no row by exact id or as a plan row exactly one row declares |
| `board-row-ambiguous` | the token is a plan row id declared by more than one row |
| `board-row-closed` | the resolved row is `Done` or `Canceled` |
| `board-row-not-in-commit` | the resolved row id appears in no added line of the staged change |
| `board-declaration-conflict` | the message and the committed change name different rows |
| `board-fetch-page-truncated` | a page at the requested size with no continuation and no declared total |
| `board-fetch-count-unstable` | two enumerations disagree on the row count or the row id set |
| `board-fetch-routes-disagree` | a milestone lists a row the project listing does not return |
| `board-fetch-paging-unsupported` | a continuation key with no known request field, or a cursor loop, or the page cap |
| `board-fetch-key-missing` | `OPERATELY_API_KEY` is not in the environment |
| `board-snapshot-unstamped` | a capture with no `fetch.fetched_at`, no integer `fetch.task_count`, or a count that disagrees with its rows |
| `board-snapshot-stale` | a capture older than the staged change (the git index mtime) |

## The board this is resolved against

The row must be resolvable from an **enumerated** board. Measured on this
instance (external v1 API, `2026-10-04`):

| route | result |
|---|---|
| `tasks/list?project_id=<id>` | 95 rows apeiron, 54 song, 0 penglai; `project_id` required; payload keys `['tasks']` only |
| `projects/list_milestone_tasks` | 26 apeiron rows, a strict subset of the 95 |
| `spaces/list_tasks?space_id=<id>` | 14 rows, none of them in apeiron's 95 |
| `projects/list_tasks` | `HTTP 404 Query not found` |
| `spaces/list_tasks` paging inputs | `limit`, `cursor`, `offset`, `per_page`, `take`, `count`, `after`, `first`, `page` rejected as `Unknown input field`; `page_size`, `starting_after`, `items_per_page`, `max_results` accepted by schema and answered `HTTP 500` |

No route pages on this instance, so completeness is not read off a single
response. It is the conjunction of three independent signals, and the capture
names which carried the argument: the cap (`rows < requested page size`, or a
declared total met), the stability of two enumerations, and cross-route
agreement. A pre-fix capture resolved against **77 of 95** rows, which is why a
live row read as `board-row-not-found` and the commit declared an older row
instead.

## Not covered

The link records that a commit declared a row. It does not move the row: there
is no move-task route over REST and no MCP call from a hook, so moving the row
remains the lead's action at the board. `board.py`'s `normalize_document` does not
yet carry a `commits` array, so the link is read back by `tools/hooks/lane.py`'s
own reader (`commit_links`), which is the gate named above; folding `commits`
into the driver is a separate task in `tools/board/board.py`.
---

Row: `APEIRON-COMMIT-ONTOLOGY` (`apeiron-commit-ontology-GgR4fRbaXs8jOjmHOLyCF5`).
