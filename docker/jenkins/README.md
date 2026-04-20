# FINN Jenkins CI — maintainer guide

Every parallel stage is defined by **one row** of `PARALLEL_SHARDS` at the
top of [`Jenkinsfile`](./Jenkinsfile). Actual test selection and shard
splitting are done by the standard `-m <marker>` expression plus the tiny
`--num-shards` / `--shard-id` plugin in [`tests/conftest.py`](../../tests/conftest.py).
No state, no allowlists, no rebalancing script.

The HW-tethered [`Jenkinsfile_HW`](./Jenkinsfile_HW) uses the same pattern
with a parallel `HW_SHARDS` table for per-board shards.

## How do I …

### … add a new test?

Decorate it with the existing marker (`@pytest.mark.fpgadataflow`,
`@pytest.mark.end2end`, `@pytest.mark.bnn_u250`, …). That is the entire CI
change. The next CI run deterministically hashes the new test's `nodeid`
into one of the shards for that marker.

### … add a new BNN parameter value (e.g. a new `(wbits, abits)` combo)?

Edit the lists in `tests/end2end/test_end2end_bnn_pynq.py` (the `_BNN_WBITS`
/ `_BNN_ABITS` / `_BNN_TOPOLOGY` constants). Nothing else. The new parameter
lands in some shard of the board's marker automatically.

### … add a new BNN board?

1. Add the marker (`bnn_<board>`) to `setup.cfg` under `[tool:pytest]` `markers`.
2. Add a line to `_BNN_MARKER_BY_BOARD` in
   `tests/end2end/test_end2end_bnn_pynq.py` and a matching entry in
   `test_board_map` (`src/finn/util/basic.py`).
3. Add one row to `BNN_BOARD_ROWS` in `Jenkinsfile` — marker, shard count,
   worker count, board name. `PARALLEL_SHARDS` and `BNN_BOARDS` are both
   derived so nothing else changes.
4. (HW pipeline only.) Adding a board is a **three-place edit** in
   `Jenkinsfile_HW`:
   a. Add a row to `HW_SHARDS` with `board`, `label`, `onlineEnv`,
      `credentialsId`, `testType`, and `restartPrep` fields.
   b. Add a matching arm to the static `isOnline(onlineEnvName)` `switch`
      — `switch` is used (not dynamic `env.getProperty()`) to stay inside
      the Groovy sandbox whitelist. A missing arm raises a clear runtime
      `error`.
   c. Add a matching static assignment in `refreshNodeOnlineFlags()`
      (`env.<ONLINE_ENV> = isNodeOnline('<label>') ? 'true' : 'false'`).
   There is no `validateShards` equivalent for `HW_SHARDS`; a missed edit
   only fails at runtime when the board first comes online.

### … debug one stage without running the whole pipeline?

Trigger a build with `STAGES=<substring>` set on the `Jenkinsfile` job.
`eachActiveShard` skips rows whose stage name does not contain the
substring. Combine with the boolean params (`sanity` / `fpgadataflow` /
`end2end`) as usual. Example: `STAGES='Sanity - Unit Tests'` to run only
the sanity unit shard; `STAGES='BNN U250'` to run only the U250 row.

### … pin a flaky test to a specific shard?

Decorate the test with `@pytest.mark.shard(N)` where `0 <= N < num_shards`
for that marker. The sharding hook honours the pin instead of hashing. Use
this for test isolation during debugging; remove once the flake is fixed so
balanced hashing resumes.

**Trap:** if the pinned test shares an `@pytest.mark.xdist_group(...)` with
siblings that hash to a different shard, the chain splits and downstream
`load_test_checkpoint_or_skip` calls silently skip. Either pin *every* test
in the group to the same `N`, or only pin tests that are already in a
singleton group (e.g. top-of-chain tests like `test_export`). The hook
validates that `N` is in range, but cannot see `xdist_group` membership.

### … preview a shard split before pushing a `shards:` bump?

Run the pytest plugin in dry-run mode locally:

```bash
pytest --collect-only --num-shards 4 --dry-run-shards -m '<marker>'
```

It prints a `shard | count | sample_nodeid` table and exits without
running any tests.

### … handle a test that got much slower?

Check the per-shard timing summary at the top of `Check Stage Results` in
the console log — `aggregateReports()` prints a
`stash | id | wall_s | max_group_s | max_group` table and flags any shard
exceeding 1.5x the family median. `reports/<stash>.timings.json` is also
archived for offline analysis. If a whole marker's wall-clock grows out
of budget, bump its `shards:` count in `PARALLEL_SHARDS`.

### … read the timing JSON directly?

Each shard emits `reports/<stash>.timings.json`:

```json
{
  "stash": "fpgadataflow_1",
  "shard": {"num": 2, "id": 0},
  "wall_seconds": 1843.2,
  "total_test_seconds": 7123.4,
  "groups": [
    {"name": "<xdist_group>", "count": 18, "seconds": 982.1},
    ...
  ]
}
```

### … find which stage runs a given test?

Run `pytest --collect-only -m '<marker>' <path/to/test.py>` locally. The
marker is what CI selects on. The Jenkins console log also echoes
`runPytest[<stash>]: FINN_CI_MARKER='<marker>'; python -m pytest …` at
the start of every stage.

### … verify a marker still has CI coverage?

Any marker used in `PARALLEL_SHARDS` that collects **zero** tests will fail
the stage loudly — the plugin raises `UsageError`. A silent-skip is
impossible by construction. At pipeline start, `validateShards()` also
checks every row's marker against `^[A-Za-z0-9_ ]+$` and fails fast if a
marker could inject shell metacharacters.

Note: iter-5 extends this guard to **unsharded rows** too (they run with
`--num-shards=1 --shard-id=0` so timings are uniform). Markers that were
legitimately empty in iter-4 used to exit 5 → 0 silently; they now hard-fail
at collection. If you see a new "no tests collected for this marker"
failure after iter-5, this is why.

## Stage → param mapping

Stages are gated by the job parameters `sanity`, `fpgadataflow`, `end2end`,
and optionally filtered further by `STAGES` (substring match on stage
name). The `Sanity - Build Hardware` row is suppressed when `end2end=true`
because the BNN rows rebuild the same scenarios.

## Pre-flight validation

The first stage of every build is `Validate`, which runs `validateShards()`:

- Every `PARALLEL_SHARDS` row's `marker` matches `^[A-Za-z0-9_ ]+$`.
- `shards` is a positive integer.
- `BNN_BOARDS` is consistent with `BNN_BOARD_ROWS` (drift check).
- The total active shard count across all enabled params doesn't exceed
  the `finn-build` label's total executors — UNSTABLE (not FAILURE) if
  over-budget, since CI queues correctly either way but wall-clock will
  extend silently.

Per-row marker regex violations are a hard `error`; over-budget is
`unstable`. Fix by tightening the marker expression or reducing `shards:`
values.

## Artefacts

- `reports/*.xml`, `reports/*.html` (merged via `pytest_html_merger`)
- `reports/<stash>.timings.json` per shard (per-group wall-clock)
- `coverage_<stash>/` per row with `coverage: true`
- `<Board>.zip` per row with `zipBoards: [...]`.
  `assertZipBoardsEmitted()` in `aggregateReports()` verifies at least one
  of the expected zips exists per active row and emits `UNSTABLE` with a
  clear diagnostic if a row's bitstream-producing scenario was not
  executed (e.g. hash-sharding happened to skip the cnv-w2a2 scenario on
  every shard).

## HW pipeline

[`Jenkinsfile_HW`](./Jenkinsfile_HW) runs the board-tethered tests. It is
structured around `HW_SHARDS` (one row per board) and `HW_TEST_TYPES`
(`bnn_build_sanity` + `bnn_build_full`). Offline boards are gated at stage
entry via the `isNodeOnline(label)` helper called in
`refreshNodeOnlineFlags()` — the pipeline never enters `node(offlineLabel)`
so offline boards cannot hang the build.

HW aggregation uses the same `aggregateReports()` helper shape as the
main `Jenkinsfile` (junit FIRST, then defensive `archiveArtifacts`). The
helpers (`safeStashReport`, `aggregateReports`, `cleanPreviousBuildFiles`,
`unstashIfPresent`) are **duplicated** verbatim between the two files —
Jenkins shared libraries are not available in this repo. A `// keep in
sync` comment at the top of each duplicated block flags this.

Changes to iter-5's `Jenkinsfile` (SW CI) are validated by
`finn_ci_speedup #17`; changes to `Jenkinsfile_HW` require a separate
`finn-hw` build with at least one board online.
