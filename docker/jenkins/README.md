# FINN Jenkins CI maintainer guide

Every parallel stage is defined by one row of `STAGES` in [`ci_sharding.py`](./ci_sharding.py). The [`Jenkinsfile`](./Jenkinsfile) loads that list during Validate with `python3 docker/jenkins/ci_sharding.py stages-json`, and the pytest plugin in [`tests/conftest.py`](../../tests/conftest.py) imports the same module for shard assignment and `--which-shard` lookup. Test selection is the standard `-m <marker>` expression, and shard splitting is the `--num-shards` / `--shard-id` plugin.

The HW-tethered [`Jenkinsfile_HW`](./Jenkinsfile_HW) follows the same broad pattern with a separate `HW_SHARDS` table.

## Dynamic timing state

Validate prepares a per-build timing snapshot from `${FINN_NFS_ROOT_BASE}/_ci_state/<jobKey>/ci_timings_master.json`. Each shard copies that snapshot into its workspace before launching Docker and exports the workspace copy to pytest as `FINN_CI_TIMINGS_FILE`. If no master exists, it is seeded from [`ci_timings_seed.json`](./ci_timings_seed.json); if the NFS root or snapshot is unavailable, shards read that seed directly. Each shard writes `<stash>.timings.json`, Check Stage Results merges the latest observed groups back into the master with `python3 docker/jenkins/ci_sharding.py update`, and the archived `reports/ci_timings_master.json` shows the resulting state for that build.

Fallback behaviour is deliberately non-fatal: if the NFS root is unavailable, the master file is missing, JSON parsing fails, a group is absent, or the master update cannot be written, pytest still runs. Known groups use recorded seconds, unknown groups use the median recorded weight, and a run with no useful timing signal uses deterministic round-robin assignment over sorted group keys.

Partial builds only refresh groups observed by that run. Groups not present in the current run remain in the master with their previous timing.

## Per-build host tmp

On `finn-build` agents, each run uses a per-shard host build directory at `$FINN_AGENT_NFS_ROOT/workspace/tmp/ci_runs/<jobKey>/<BUILD_NUMBER>/<stash>`.

`jobKey` is `JOB_NAME` sanitised to one path segment, so two jobs at the same `BUILD_NUMBER` cannot collide. The Validate stage runs `python3 docker/jenkins/ci_sharding.py prune-tmp` once per build. It walks every per-agent subtree under `${FINN_NFS_ROOT_BASE}/<NODE_NAME>/workspace/tmp/ci_runs/<jobKey>/`, keeps the largest N numeric build directories, always keeps the current `BUILD_NUMBER`, and removes older directories whose mtime exceeds M days. Tuning constants live at the top of `Jenkinsfile`: `FINN_CI_TMP_RETAIN_BUILDS` defaults to 5 and `FINN_CI_TMP_MAX_AGE_DAYS` defaults to 14.

## How do I ...

### Add a new test?

Decorate it with the existing marker, for example `@pytest.mark.fpgadataflow`, `@pytest.mark.end2end`, or `@pytest.mark.bnn_u250`. The next CI run picks it up automatically.

### Add a new BNN parameter value?

Edit the `_BNN_WBITS`, `_BNN_ABITS`, and `_BNN_TOPOLOGY` constants in `tests/end2end/test_end2end_bnn_pynq.py`. Nothing else is needed.

### Add a new BNN board?

1. Add the marker `bnn_<board>` to `setup.cfg` under `[tool:pytest]`.
2. Add a line to `_BNN_MARKER_BY_BOARD` in `tests/end2end/test_end2end_bnn_pynq.py` and a matching entry in `test_board_map` in `src/finn/util/basic.py`.
3. Add one row to `STAGES` in [`ci_sharding.py`](./ci_sharding.py).
4. For the HW pipeline only, add a row to `HW_SHARDS` in `Jenkinsfile_HW` with `board`, `agentLabel`, `onlineEnv`, `credentialsId`, `setupScript`, `marker`, and `restartPrep`.

### Debug one stage without running the whole pipeline?

Trigger a build with `STAGES=<substring>` set on the Jenkins job, and `eachActiveShard` skips rows whose stage name does not contain the substring. Combine with the boolean params `sanity`, `fpgadataflow`, and `end2end` as usual.

### Pin a flaky test to a specific shard?

`@pytest.mark.shard(N)` pins a test and its `xdist_group` siblings to shard N. If the pinned test shares an `xdist_group` with siblings, pin every member explicitly to avoid splitting the chain.

### Preview a shard split before pushing?

```bash
pytest --collect-only --num-shards 4 --dry-run-shards -m '<marker>'
```

This prints `shard | items | groups | weight_s | sample_group` and exits. Set `FINN_CI_TIMINGS_FILE=<path>` to preview with a synthetic or archived timing state.

### Find which stage and shard runs a given test?

For an archived Jenkins build, download or open `reports/shard_map.txt` and search for the nodeid or any useful substring. Each row is grep-friendly:

```text
nodeid=<nodeid> stage=<stage> shard=<i>/<n> stash=<stash> group=<group> weight_s=<seconds> source=<known|fallback|pinned|round_robin>
```

For local collection, use:

```bash
pytest --collect-only --which-shard test_relu_elementwisemax
```

This walks every row of `STAGES`, uses the same timing file and fallback logic as CI, and prints `stage | marker | shards | shard | stash | nodeid` rows. The `stash` column is the exact Jenkins stash name.

### Inspect or update timing state manually?

Use `python3 docker/jenkins/ci_sharding.py summarize path/to/reports/` to print per-shard wall-clock outliers from archived `<stash>.timings.json` files. Use `python3 docker/jenkins/ci_sharding.py update --reports path/to/reports --master path/to/ci_timings_master.json --out path/to/reports/ci_timings_master.json` to merge a report directory into a master file.

## Artefacts

- `reports/*.xml`, `reports/*.html` from pytest and `pytest_html_merger`.
- `reports/<stash>.timings.json` per shard.
- `reports/<stash>.shardmap.txt` and `reports/<stash>.shardmap.json` per shard.
- `reports/shard_map.txt` and `reports/shard_map.json` merged across all shards.
- `reports/ci_timings_master.json` archived copy of the updated timing state.
- `coverage_<stash>/` per row with `coverage: true`.
- `<Board>.zip` per row with `zipBoards: [...]`; `assertZipBoardsEmitted` flags the row UNSTABLE if no shard produced the expected zip.

## HW pipeline

[`Jenkinsfile_HW`](./Jenkinsfile_HW) runs the board-tethered tests. It is structured around `HW_SHARDS` and `HW_TEST_TYPES` (`bnn_build_sanity` plus `bnn_build_full`). Offline boards are gated by `isNodeOnline(label)` so the pipeline never enters `node(offlineLabel)`.

The shared helpers (`safeStashReport`, `unstashIfPresent`, `aggregateReports`, `cleanPreviousBuildFiles`) are duplicated between `Jenkinsfile` and `Jenkinsfile_HW` because Jenkins shared libraries are not available in this repo. A `// keep in sync` comment marks the duplicated block.
