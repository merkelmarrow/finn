# FINN Jenkins CI maintainer guide

Every parallel stage is defined by one row of `STAGES` in [`ci_sharding.py`](./ci_sharding.py). The [`Jenkinsfile`](./Jenkinsfile) loads that list during Validate with `python3 docker/jenkins/ci_sharding.py stages-json`, and the pytest plugin in [`tests/conftest.py`](../../tests/conftest.py) imports the same module for shard assignment and `--which-shard` lookup. Test selection is the standard `-m <marker>` expression, and shard splitting is the `--num-shards` / `--shard-id` plugin. Ordinary local pytest collection remains unchanged unless the CI sharding options are used.

The HW-tethered [`Jenkinsfile_HW`](./Jenkinsfile_HW) follows the same broad pattern with a separate `HW_SHARDS` table.

## Dynamic timing state

Validate prepares a per-build timing snapshot from `${FINN_NFS_ROOT_BASE}/_ci_state/<jobKey>/ci_timings_master.json`. Each shard copies that snapshot into its workspace before launching Docker and exports the workspace copy to pytest as `FINN_CI_TIMINGS_FILE`. If no master exists, it is seeded from [`ci_timings_seed.json`](./ci_timings_seed.json). If the NFS root or snapshot is unavailable, shards read that seed directly. Each CI row, including `1/1` rows, runs with `--num-shards` so pytest writes `<stash>.timings.json`. Check Stage Results always writes an archived `reports/ci_timings_master.json` preview, but the persistent master is updated only for full successful `sanity + fpgadataflow + end2end` runs or when `promote_ci_timings=true` is set.

Fallback behaviour is deliberately non-fatal: if the NFS root is unavailable, the master file is missing, JSON parsing fails, a group is absent, or the master update cannot be written, pytest still runs. Known groups use recorded seconds, unknown groups use the median recorded weight, and a run with no useful timing signal uses deterministic round-robin assignment over sorted group keys.

Partial builds only refresh groups observed by that run in the archived preview. Groups not present in the current run remain in the preview with their previous timing, and the persistent master is left untouched unless the run is explicitly promoted.

## SW-to-HW zip handoff

The SW pipeline stages board deployment directories per shard, then Check Stage Results aggregates those staged deployments into one board bitstream zip plus a sibling `.READY` marker in the canonical per-build directory:

    ${ARTIFACT_DIR}/ci_runs/<jobKey>/<BUILD>/
      zips/<hwTestType>/<board>.zip
      zips/<hwTestType>/<board>.zip.READY
      BUILD_INFO.txt
      deployments/<hwTestType>/<board>/<stash>/<board>/<model>/

The `.READY` marker is the SW-to-HW handshake. It is touched only after the aggregated zip has been renamed into place, so a half-written zip or an aborted shard never leaves a READY pointing at incomplete bytes. HW resolves each `(testType, board)` pair independently to the newest build under `ci_runs/<jobKey>/` whose `<board>.zip.READY` sibling is present. A board whose build failed has no READY this build, so HW can fall back to that board's previous READY. Fallback is measured against the newest numeric SW build directory, not just the newest selected READY zip, and marks the HW build unstable unless `allow_fallback=true` is set on the HW job. There is no global READY marker and no build-wide "is this build good?" decision. The handoff itself has no JSON schema, no version field, no symlink, the directory layout is the contract. The persistent timing master is JSON, but that lives in a separate `_ci_state/` tree under `FINN_NFS_ROOT_BASE` and is not part of the SW-to-HW channel.

`ARTIFACT_DIR` is required for SW runs that select rows with `zipArtifacts` and for the HW pipeline, the per-build tree is the shared filesystem channel between SW and HW. Software-only debug runs can run without it. A `STAGES` row that produces these zips declares a `zipArtifacts` nested key:

```python
"zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["U250"]}
```

`hwTestType` (today `bnn_build_sanity` or `bnn_build_full`) selects which HW pipeline category the zip feeds. `boards` is the list of board zips the row produces. The nested shape means the pair is either present or absent, so no separate pairing assertion is needed in Validate.

Operators pin to a specific SW build for debugging by setting the `sw_build_dir` parameter on the HW job (full path to a single SW build directory, forces every board through that one directory). `sw_job_name` (default `finn`) selects which job's `ci_runs/<jobKey>/` tree to scan for per-board READY zips.

`BUILD_INFO.txt` is plain `KEY=VALUE` per line. It is never parsed by code, just `cat` it to find out the job/build/commit/branch/date/enabled-params/timings-snapshot/docker-image-dir/validate-node for any given build directory. The HW pipeline archives the BUILD_INFO.txt of every distinct SW build it pulled a zip from, named `sw_build_info_<N>.txt`, and lists the per-board source builds in the HW build description.

Promotion of the per-build timing preview into the persistent timing master is controlled by the `promote_ci_timings` Jenkins parameter (default off) on the SW job. Auto-promote also fires implicitly when every row in `STAGES` would actually execute on this run (every `param` enabled, no `skipWhen` row blocked, and `STAGE_FILTER` empty). The canonical `STAGES` has no `skipWhen` rows, so a successful build with `sanity + fpgadataflow + end2end` all ticked refreshes the master automatically. Setting `promote_ci_timings=true` is reserved for forcing a refresh from a partial run (e.g. a single stage rerun after a one-off failure).

## Per-build storage and retention

Set `FINN_CI_NFS_ROOT` once on the Jenkins controller and the SW pipeline derives every shared subtree from it. There is nothing else to set up. The conventional layout below is what the resolvers produce. If you ever need to relocate a specific tree (e.g. point the docker image cache at a faster mount), set the corresponding legacy env var alongside `FINN_CI_NFS_ROOT` and the legacy value wins. Validate echoes a one-line `warnLegacyNfsEnv` reminder when a legacy override is active so the operator knows which path actually took effect.

| Tree | Path under `FINN_CI_NFS_ROOT` | Legacy override | Subcommand | Policy |
| --- | --- | --- | --- | --- |
| Per-shard host tmp | `agent_workspaces/<NODE>/workspace/tmp/ci_runs/<jobKey>/<BUILD>/<stash>` | `FINN_NFS_ROOT_BASE` | `prune-tmp` | `TRANSIENT_RETENTION` (retain=5, ageDays=14) |
| Shared Docker image | `docker_images/<jobKey>/<BUILD>/` | `FINN_DOCKER_SHARED_DIR` | `prune-images` | retain=3, ageDays=14 |
| SW-to-HW handoff | `artifacts/ci_runs/<jobKey>/<BUILD>/` | `ARTIFACT_DIR` | `prune-artifacts` | `HANDOFF_RETENTION` (retain=30, ageDays=30) |
| Timing master + snapshots | `agent_workspaces/_ci_state/<jobKey>/` | (no override) | n/a | n/a |

`FINN_AGENT_NFS_ROOT` keeps its independent role as a per-agent override (local-SSD escape hatch) and is unaffected by this consolidation.

When a shared Docker image directory is configured, non-build stages run `run-docker.sh` with `FINN_DOCKER_PREBUILT=1`. In that mode `run-docker.sh` treats `FINN_DOCKER_SHARED_IMAGE_DIR` as authoritative, always verifies or loads that build-scoped image even when a same-tag local image exists, and fails fast if the shared image is missing, unusable, or tagged differently. A local fallback build is available only when `FINN_DOCKER_ALLOW_PREBUILT_FALLBACK=1` is set deliberately for developer recovery.

Validate rotates all three numeric-keyed trees via the single `rotateBuildTrees()` helper in [`Jenkinsfile`](./Jenkinsfile). Each rotation keeps the newest N numeric subdirs and the current build, and deletes older subdirs whose mtime exceeds M days. All three subcommands skip silently when their parent directory does not exist, and the Python side tolerates concurrent rmtree races (a second CI run pruning the same parent will not abort the rotation).

`jobKey` is `JOB_NAME` sanitised by `ci_sharding.job_key()` to one path segment, leading/trailing dots stripped so `JOB_NAME=".."` cannot escape into the parent directory. Both Jenkinsfiles call the Python helper via `python3 docker/jenkins/ci_sharding.py job-key <name>` so the sanitisation rule has one source of truth. The retention values live in the `RETENTION` dict at the top of [`ci_sharding.py`](./ci_sharding.py), and the SW pipeline loads them via `python3 docker/jenkins/ci_sharding.py retention-json` during Validate, so tuning is a one-file change.

Artifact-tree pruning is safe because HW resolves per board to the newest `.READY` zip remaining. Deleting an older build directory just makes HW fall through to the next-oldest READY on its next collect pass. Artifact retention is sized to outlast the longest single-board failure streak; HW falling back to a build older than this window indicates a separate problem (not a tuning issue).

`prune-tmp` walks every per-agent subtree under `${FINN_NFS_ROOT_BASE}/<NODE_NAME>/workspace/tmp/ci_runs/<jobKey>/` and reports the aggregate `matched=N` count across all of them, so a single log line shows whether anything was actually removed.

A corrupt master timing file is renamed aside (`ci_timings_master.json.corrupt-<epoch>`) before the in-progress run writes its replacement, so a one-off NFS hiccup never loses the historical timing data silently.

## How do I ...

### Add a new test?

Decorate it with the existing marker, for example `@pytest.mark.fpgadataflow`, `@pytest.mark.end2end`, or `@pytest.mark.bnn_u250`. The next CI run picks it up automatically.

### Add a new BNN parameter value?

Edit the `_BNN_WBITS`, `_BNN_ABITS`, and `_BNN_TOPOLOGY` constants in `tests/end2end/test_end2end_bnn_pynq.py`. Nothing else is needed.

### Add a new BNN board?

1. Add the marker `bnn_<board>` to `setup.cfg` under `[tool:pytest]`.
2. Add a line to `_BNN_MARKER_BY_BOARD` in `tests/end2end/test_end2end_bnn_pynq.py` and a matching entry in `test_board_map` in `src/finn/util/basic.py`.
3. In [`ci_sharding.py`](./ci_sharding.py), add a `BOARDS` entry with `agentLabel`, `credentialsId`, `restartPrep`, `setupScript`, `marker`, and a `STAGES` row that references the board in its `zipArtifacts.boards`. `validate_boards()` cross-checks the two on every CLI invocation. `Jenkinsfile_HW` derives `HW_SHARDS` from `BOARDS` via `python3 docker/jenkins/ci_sharding.py hw-shards-json`, no separate edit needed.

### Trigger a build for a contribution PR?

The `PRESET` choice param drives the matrix:

- `PRESET=custom` (default) honours the individual boolean params verbatim. The defaults (`sanity=true`, `fpgadataflow=false`, `end2end=false`) give a sanity-only build, the same as the pre-PRESET default. Tick `fpgadataflow` or `end2end` to add those rows.
- `PRESET=smoke` is an explicit "sanity only, ignore my booleans" override. Pick this when you want a quick smoke regardless of what the booleans are set to.
- `PRESET=full` runs `sanity + fpgadataflow + end2end` (booleans ignored) and auto-promotes the timing master on success. This is the nightly setting.

### Debug one stage without running the whole pipeline?

Trigger a build with `PRESET=custom`, the boolean param for the relevant family ticked, and `STAGE_FILTER=<substring>` set so `eachActiveShard` skips rows whose stage name does not contain the substring.

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

Use `python3 docker/jenkins/ci_sharding.py summarize path/to/reports/` to print per-shard wall-clock outliers from archived `<stash>.timings.json` files. Use `python3 docker/jenkins/ci_sharding.py update --reports path/to/reports --master path/to/ci_timings_master.json --out path/to/reports/ci_timings_master.json` to write a non-promoted preview, or add `--promote --full-run --job <job> --build <build>` to update the persistent master.

### Refresh the seed file from a promoted master?

`docker/jenkins/ci_timings_seed.json` is the fallback timing snapshot used when the persistent master is unreachable (cold-start agent, NFS hiccup, source-tree-only environment). Refresh it from a known-good master with:

```bash
python3 docker/jenkins/ci_sharding.py regen-seed \
  --master "${FINN_NFS_ROOT_BASE}/_ci_state/finn/ci_timings_master.json" \
  --out docker/jenkins/ci_timings_seed.json
```

Commit the result alongside any test set change that introduced or renamed groups.

## Artefacts

- `reports/*.xml`, `reports/*.html` from pytest and `pytest_html_merger`.
- `reports/<stash>.timings.json` per shard.
- `reports/<stash>.shardmap.txt` and `reports/<stash>.shardmap.json` per shard.
- `reports/shard_map.txt` and `reports/shard_map.json` merged across all shards.
- `reports/ci_timings_master.json` archived copy or preview of timing state, with `last_update.promoted` recording whether the persistent master was changed.
- `coverage_<stash>/` per row with `coverage: true`.
- `${ARTIFACT_DIR}/ci_runs/<jobKey>/<BUILD_NUMBER>/zips/<hwTestType>/<board>.zip` per row with a `zipArtifacts` entry. `aggregateReports()` runs `assertZipArtifactsEmitted()` which marks the build UNSTABLE (non-fatal) when an active row declared `zipArtifacts` but no `.READY` was written.
- `${ARTIFACT_DIR}/ci_runs/<jobKey>/<BUILD_NUMBER>/zips/<hwTestType>/<board>.zip.READY` per-board handshake marker, touched only after the zip is in place. Publishing is idempotent for same-build retries: a replay rewrites the zip atomically and leaves or refreshes READY instead of requiring a manual tree wipe. HW resolves per board to the newest build with this marker.
- `${ARTIFACT_DIR}/ci_runs/<jobKey>/<BUILD_NUMBER>/BUILD_INFO.txt` for human traceability.

## HW pipeline

[`Jenkinsfile_HW`](./Jenkinsfile_HW) runs the board-tethered tests. It is structured around `HW_SHARDS` and `HW_TEST_TYPES` (`bnn_build_sanity` plus `bnn_build_full`). Offline boards are gated by `isNodeOnline(label)` so the pipeline never enters `node(offlineLabel)`, and an offline board with a READY zip marks the HW build unstable instead of silently passing. Board+testType pairs with no READY zip resolved by `resolveSwBoardZipPaths()` are skipped, so HW jobs run cleanly against partial software-CI runs and a regressed single board does not block the others.

The Collect stage populates a `SW_BOARD_ZIP_PATHS` map keyed by `<testType>_<board>`. Every later step (`buildHwStageMap`, `stashBuildArtifacts`, `expectedStashes`) is a pure map lookup, no filesystem stats run outside the Collect node.

SW-build resolution is automatic via `sw_job_name` (default `finn`), see "SW-to-HW zip handoff" above. `sw_build_dir` is an optional explicit override (full path to a single SW build directory) for off-Jenkins recovery or pinning a specific build.

Board zips are retrieved by direct filesystem read from each board's resolved SW build directory on the `finn-build` aggregator agent, then `stash`/`unstash`'d to the board agents. Boards whose zip has no `.READY` sibling are skipped from the per-board stash and per-shard branch maps. If a whole HW stage has no READY zips, the stage marks the build unstable with a clear message instead of calling `parallel` with an empty branch map.

Shared Groovy helpers (`paramBool`, `paramString`, `shellQuote`, `safeStashShardReport` / `safeStashHwReport`, `unstashIfPresent`, `cleanPreviousBuildFilesStrict` / `cleanPreviousBuildFilesHw`) live in [`_common.groovy`](./_common.groovy). Each Jenkinsfile loads it once via `load 'docker/jenkins/_common.groovy'` and exposes thin top-level wrappers so call sites stay readable. `safeStash*` and `cleanPreviousBuildFiles*` are split into two names because the SW and HW pipelines genuinely disagree on what to include/exclude (HW takes an explicit `fileBase` and stashes only XML+HTML, SW stashes the full sidecar set; HW also sudos and optionally sweeps a sibling `.zip`, SW hard-fails on root-owned residue and recreates the directory). `aggregateReports` stays inline in each Jenkinsfile because the SW form does many more things than the HW form.
