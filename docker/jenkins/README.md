# FINN Jenkins CI — maintainer guide

Every parallel stage is defined by **one row** of `STAGES` in
[`tests/ci_shards.py`](../../tests/ci_shards.py). The
[`Jenkinsfile`](./Jenkinsfile) loads that list at the Validate stage into
`PARALLEL_SHARDS` (via `python3 -c ... readJSON`); the pytest plugin in
[`tests/conftest.py`](../../tests/conftest.py) reads it for the
`--which-shard` lookup. Test selection is the standard `-m <marker>`
expression; shard splitting is the `--num-shards` / `--shard-id` plugin.

The HW-tethered [`Jenkinsfile_HW`](./Jenkinsfile_HW) follows the same
pattern with a parallel `HW_SHARDS` table.

## Per-build host tmp (`ci_runs`) and rotation

On `finn-build` agents, each run uses a per-shard host build directory at:

`$FINN_AGENT_NFS_ROOT/workspace/tmp/ci_runs/<jobKey>/<BUILD_NUMBER>/<stash>`

`jobKey` is `JOB_NAME` sanitised to one path segment (so e.g. `finn` and
`finn_ci_speedup` at the same `BUILD_NUMBER` cannot collide). Stash names
are unchanged.

The Validate stage runs
[`scripts/rotate_finn_ci_tmp.sh`](../../scripts/rotate_finn_ci_tmp.sh)
once per build. It keeps the largest N numeric build directories under
`…/ci_runs/<jobKey>/`, always keeps the current `BUILD_NUMBER`, and
removes directories older than M days. Tuning constants live at the top
of `Jenkinsfile`: `FINN_CI_TMP_RETAIN_BUILDS` (default 5) and
`FINN_CI_TMP_MAX_AGE_DAYS` (default 14). Pass `--dry-run` to the script
to preview deletions.

## How do I …

### … add a new test?

Decorate it with the existing marker (`@pytest.mark.fpgadataflow`,
`@pytest.mark.end2end`, `@pytest.mark.bnn_u250`, …). The next CI run picks
it up automatically.

### … add a new BNN parameter value (e.g. a new `(wbits, abits)` combo)?

Edit the `_BNN_WBITS` / `_BNN_ABITS` / `_BNN_TOPOLOGY` constants in
`tests/end2end/test_end2end_bnn_pynq.py`. Nothing else.

### … add a new BNN board?

1. Add the marker (`bnn_<board>`) to `setup.cfg` under `[tool:pytest]`.
2. Add a line to `_BNN_MARKER_BY_BOARD` in
   `tests/end2end/test_end2end_bnn_pynq.py` and a matching entry in
   `test_board_map` (`src/finn/util/basic.py`).
3. Add one row to `STAGES` in
   [`tests/ci_shards.py`](../../tests/ci_shards.py).
4. (HW pipeline only.) In `Jenkinsfile_HW`:
   - add a row to `HW_SHARDS` with `board`, `agentLabel`, `onlineEnv`,
     `credentialsId`, `setupScript`, `marker`, `restartPrep`;
   - add a matching arm to `isOnline(onlineEnvName)` (a static `switch`
     so we stay inside the Groovy sandbox whitelist);
   - add a matching `env.<ONLINE_ENV> = isNodeOnline(...)` line in
     `refreshNodeOnlineFlags()`.

### … debug one stage without running the whole pipeline?

Trigger a build with `STAGES=<substring>` set on the Jenkins job;
`eachActiveShard` skips rows whose stage name doesn't contain the
substring. Combine with the boolean params (`sanity` / `fpgadataflow` /
`end2end`) as usual.

### … pin a flaky test to a specific shard?

`@pytest.mark.shard(N)` pins a test (and its xdist_group siblings) to
shard N. If the pinned test shares an `xdist_group` with siblings that
hash elsewhere, pin every member explicitly to avoid splitting the
chain.

### … preview a shard split before pushing?

```bash
pytest --collect-only --num-shards 4 --dry-run-shards -m '<marker>'
```

Prints `shard | items | groups | weight_s | sample_group` and exits.

### … find which stage and shard runs a given test?

```bash
pytest --collect-only --which-shard test_relu_elementwisemax
```

Walks every row of `STAGES`, reuses the canonical
`_assign_groups_to_shards()` from `conftest.py`, and prints
`stage | marker | shards | shard | stash | nodeid` rows. The `stash`
column is the exact Jenkins stash name; pass it as `STAGES=<substring>`
to re-run only that shard.

## Calibrating shard balance

Group → wall-seconds for the LPT-greedy bin packing in
`_assign_groups_to_shards()` is checked in at
[`tests/ci_timings.json`](../../tests/ci_timings.json). Regenerate after
test set changes that materially shift wall-clock:

```bash
# pull *.timings.json artefacts from a recent green build, then:
python3 scripts/regen_ci_timings.py path/to/reports/
```

`Check Stage Results` runs
[`scripts/summarize_ci_timings.py`](../../scripts/summarize_ci_timings.py)
on every build; shards that exceed 1.5x their family's median wall-clock
are flagged loudly so it's obvious when the file is stale. Bumping
`shards:` in `tests/ci_shards.py` is still valid for permanent capacity
changes.

## Artefacts

- `reports/*.xml`, `reports/*.html` (merged via `pytest_html_merger`)
- `reports/<stash>.timings.json` per shard
- `coverage_<stash>/` per row with `coverage: true`
- `<Board>.zip` per row with `zipBoards: [...]`. `assertZipBoardsEmitted`
  flags the row UNSTABLE if no shard produced the expected zip.

## HW pipeline

[`Jenkinsfile_HW`](./Jenkinsfile_HW) runs the board-tethered tests. It
is structured around `HW_SHARDS` (one row per board) and
`HW_TEST_TYPES` (`bnn_build_sanity` + `bnn_build_full`). Offline boards
are gated by `isNodeOnline(label)` so the pipeline never enters
`node(offlineLabel)`.

The shared helpers (`safeStashReport`, `unstashIfPresent`,
`aggregateReports`, `cleanPreviousBuildFiles`) are duplicated between
`Jenkinsfile` and `Jenkinsfile_HW` because Jenkins shared libraries are
not available in this repo. A `// keep in sync` comment marks the
duplicated block.
