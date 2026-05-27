# FINN Jenkins CI maintainer guide

## TL;DR for contributors

- Run a PR build: click *Build with Parameters*, leave `STAGES=sanity`, hit *Build*.
- Debug just one family: set `STAGES=fpgadataflow` or `STAGES=end2end`.
- Find which shard a test runs on: `pytest --collect-only --which-shard <substring>` locally, or grep `reports/shard_map.txt` on an archived build. From a finn-less checkout, `python3 docker/jenkins/ci_sharding.py which-shard <group> [--marker <marker>] [--timings <path>]` returns an approximation that uses only the timing master.
- Add a new test: decorate it with the existing marker (e.g. `@pytest.mark.fpgadataflow`). The next CI run picks it up.

The rest of this doc is the maintainer-facing detail behind those four lines.

## Where the knobs live

`FINN_CI_NFS_ROOT` is the only env var operators **must** set, in the Jenkins job DSL. Everything else derives from it. When unset the pipeline still runs end-to-end and produces pass/fail per shard, with the following features degraded: no shared Docker image cache (each agent rebuilds locally), no build-to-HW artifact handoff (the HW pipeline cannot test against this build), no persistent timing master (sharding falls back to deterministic round-robin), no per-agent NFS caches. Any build that would normally publish board artifacts marks itself UNSTABLE at aggregate time rather than failing the whole build. HW jobs hard-fail without `FINN_CI_NFS_ROOT` because they have nothing to read otherwise. Wire it into the job DSL as a global environment variable; the reference shape is a single `environmentVariables { env('FINN_CI_NFS_ROOT', '...') }` block.

Optional operator overrides, all with sensible defaults so nothing else needs to be set on a fresh install:

| Env var | Defaults to | What it changes |
| --- | --- | --- |
| `FINN_LOCAL_BUILD_LABEL` | `finn-build` | Agent label for the optional non-Docker `setup-local.sh` stage. Override only when that stage needs an apt + python3.10 host distinct from the main build pool. |
| `FINN_CI_LOCAL_CACHE_ROOT` | `${WORKSPACE_TMP}/finn-ci-cache` | Pip + XDG cache root for the same optional stage. Override on agents whose home or scratch layout matters. |
| `FINN_LSF_NFS_STAGING` | unset (no LSF tails) | When set, `archive_failure_logs.sh` includes LSF staging-dir tails in the failure bundle. |
| `FINN_DOCKER_TAG` | repo-derived | Docker image tag the build, run, and HW pipelines pull. Override only for pinned-image debugging. |

Every parallel stage is defined by one row of `STAGES` in [`ci_sharding.py`](./ci_sharding.py). The [`Jenkinsfile`](./Jenkinsfile) loads the entire config bundle during Validate with a single `python3 docker/jenkins/ci_sharding.py validate-config --choice <STAGES> --job-name <JOB_NAME>` call (which runs `validate_config()` first), and the pytest plugin in [`tests/conftest.py`](../../tests/conftest.py) imports the same module for shard assignment and `--which-shard` lookup. Test selection is the standard `-m <marker>` expression, and shard splitting is the `--num-shards` / `--shard-id` plugin. Local pytest collection is unchanged unless the CI sharding options are used.

The HW-tethered [`Jenkinsfile_HW`](./Jenkinsfile_HW) follows the same broad pattern with a separate `HW_SHARDS` table derived from `BOARDS`.

## Dynamic timing state

There is no checked-in seed file. Cold start (master absent) and snapshot-unreachable both fall through to deterministic round-robin shard assignment, so the pipeline keeps working without persistent timings. The persistent master is refreshed only by trusted full-matrix builds (`STAGES=full`, no `STAGE_FILTER`, successful build). Partial sanity/debug builds still write an archived `reports/ci_timings_master.json` preview, but do not update the shared master or advance garbage collection.

The master schema is `{"schema_version": 1, "groups": {<name>: {"samples": [last MAX_SAMPLES observations], count, consecutive_rejections, last_seen_*}}}`. Per-group weights consumed by the bin packer are the median of `samples`, so a single anomalous observation only moves one slot in a five-element window and the median is unaffected.

Two layers of anomaly protection live in `ci_sharding.update_master`:

1. **Per-group rolling median + outlier rejection.** A new observation must fall within `[OUTLIER_LOW_RATIO, OUTLIER_HIGH_RATIO]` times the current median to be accepted. The crash-floor rule additionally rejects suspiciously-low observations against historically-large medians (the "shard crashed before any real test ran" pattern). Trusted full-matrix updates can force-accept after `FORCE_ACCEPT_AFTER` rejections so real regressions eventually propagate.
2. **Build-wide anomaly veto.** When at least `MIN_ELIGIBLE_FOR_ANOMALY_VETO` observations have a prior median and more than `BUILD_WIDE_ANOMALY_RATIO` of them are out-of-band, the entire persistent update is vetoed. This catches LSF / NFS-storm days where every shard is uniformly slower and would otherwise poison every group's `consecutive_rejections` counter at once.

Garbage collection drops groups not observed in the last `GC_BUILDS_UNSEEN` trusted full-matrix updates so renamed or removed tests do not leak into the master forever. Sanity/debug builds do not advance this counter.

All tunable thresholds live at the top of [`ci_sharding.py`](./ci_sharding.py).

## Build-to-HW zip handoff

The build pipeline stages board deployment directories per shard, then Check Stage Results aggregates those staged deployments into one board bitstream zip plus a sibling `.READY` marker in the canonical per-build directory:

    ${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD>/
      zips/<hwTestType>/<board>.zip
      zips/<hwTestType>/<board>.zip.READY
      BUILD_INFO.txt
      deployments/<hwTestType>/<board>/<stash>/<board>/<model>/

The `.READY` marker is the build-to-HW handshake. It is touched only after the aggregated zip has been renamed into place, so a half-written zip or an aborted shard never leaves a READY pointing at incomplete bytes. HW resolves each `(testType, board)` pair independently to the newest build under `ci_runs/<jobKey>/` whose `<board>.zip.READY` sibling is present. A board whose build failed has no READY this build, so HW falls back to that board's previous READY. Fallback is measured against the newest numeric build directory, not just the newest selected READY zip, and marks the HW build unstable unless `allow_fallback=true` is set on the HW job. There is no global READY marker and no build-wide "is this build good?" decision. The directory layout is the contract. The persistent timing master is JSON but lives in a separate `_ci_state/` tree under `FINN_CI_NFS_ROOT` and is not part of the build-to-HW channel.

`FINN_CI_NFS_ROOT` is required by the HW pipeline and recommended for any build run that you expect to feed HW (rows with `zipArtifacts`). Builds without it still complete, with handoff steps no-oping and an UNSTABLE marker at aggregate time so HW operators see the build was not a full feeder. A `STAGES` row that produces these zips declares a `zipArtifacts` nested key:

```python
"zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["U250"]}
```

`hwTestType` (today `bnn_build_sanity` or `bnn_build_full`) selects which HW pipeline category the zip feeds. `boards` lists the board zips the row produces. The nested shape means the pair is either present or absent.

Operators pin to a specific build for debugging by setting the `build_dir` parameter on the HW job (full path to a single build directory, forces every board through that one directory). `build_job_name` (default `finn`) selects which job's `ci_runs/<jobKey>/` tree to scan for per-board READY zips.

`BUILD_INFO.txt` is plain `KEY=VALUE` per line. It is never parsed by code, so `cat` it for the job/build/commit/branch/date/enabled-params/timings-snapshot/docker-image-dir/validate-node of any given build directory. The HW pipeline archives each distinct source build's BUILD_INFO.txt as `build_info_<N>.txt` and lists the per-board source builds in the HW build description.

There is no operator-controlled promotion knob. The Jenkinsfile updates the persistent timing master only for successful full-matrix builds, while every build archives a preview for inspection.

## Per-build storage and retention

Set `FINN_CI_NFS_ROOT` once on the Jenkins controller (in the job DSL) and the build pipeline derives every shared subtree from it. There are no other CI storage env vars to set. If `FINN_CI_NFS_ROOT` is unset the pipeline runs in local fallback mode. Validate prints a banner listing the disabled features and the build completes with handoff steps no-oping. `assertZipArtifactsEmitted` marks the build UNSTABLE at aggregate time if the selected rows would normally have published board artifacts.

| Tree | Path under `FINN_CI_NFS_ROOT` | Subcommand | Retention |
| --- | --- | --- | --- |
| Per-agent caches | `agent_caches/<NODE>/{xrt,finn_cache,vivado_ip_cache}` | n/a (long-lived) | n/a |
| Shared Docker image | `docker_images/<jobKey>/<BUILD>/` | `prune-images` | retain=3, ageDays=14 |
| Build-to-HW handoff | `artifacts/ci_runs/<jobKey>/<BUILD>/` | `prune-artifacts` | retain=30, ageDays=30 |
| Per-build timing snapshots | `_ci_state/<jobKey>/build_<N>_timings_input.json` | `prune-snapshots` | retain=3, ageDays=2 |
| Timing master | `_ci_state/<jobKey>/ci_timings_master.json` | n/a (in-place updates) | n/a |

Per-shard scratch lives at `${WORKSPACE}/tmp/ci_runs/<BUILD>/<stash>` regardless of `FINN_CI_NFS_ROOT`. The workspace is per-agent (NFS-mounted via `remote_fs` on lab build hosts, local SSD elsewhere), and `git clean -ffdx` between runs handles rotation.

When a shared Docker image directory is configured, non-build stages run `run-docker.sh` with `FINN_DOCKER_PREBUILT=1`. In that mode `run-docker.sh` treats `FINN_DOCKER_SHARED_IMAGE_DIR` (set by the Jenkinsfile, not by the operator) as authoritative, always verifies or loads that build-scoped image even when a same-tag local image exists, and fails fast if the shared image is missing, unusable, or tagged differently. In local fallback mode the Jenkinsfile forces `FINN_DOCKER_PREBUILT=0` so each agent can build locally.

Validate rotates the image, artifact, and timing-snapshot trees via the single `rotateBuildTrees()` helper in [`Jenkinsfile`](./Jenkinsfile). Each rotation keeps the newest N numeric entries and the current build, and deletes older entries whose mtime exceeds M days. All three subcommands skip silently when their parent directory does not exist, and the Python side tolerates concurrent prune races.

`jobKey` is `JOB_NAME` sanitised by `ci_sharding.job_key()` to one path segment, leading and trailing dots stripped so `JOB_NAME=".."` cannot escape into the parent directory. The build pipeline reads the sanitised value out of the `validate-config` payload, and `Jenkinsfile_HW` shells out via the standalone `job-key` subcommand because it does not load the full bundle. Either way the sanitisation rule has one source of truth. Retention values live in the `RETENTION` dict at the top of [`ci_sharding.py`](./ci_sharding.py) and ride out via the same `validate-config` payload, so tuning is a one-file change.

Artifact-tree pruning is safe because HW resolves per board to the newest `.READY` zip remaining. Deleting an older build directory just makes HW fall through to the next-oldest READY on its next collect pass. Artifact retention is sized to outlast the longest realistic single-board failure streak.

A corrupt master timing file is renamed aside (`ci_timings_master.json.corrupt-<epoch>`) before the in-progress run writes its replacement, so a one-off NFS hiccup never loses the historical timing data silently.

## How do I ...

### Find which stage and shard runs a given test?

For local collection:

```bash
pytest --collect-only --which-shard test_relu_elementwisemax
```

This walks every row of `STAGES`, uses the same timing file and fallback logic as CI, and prints `stage | marker | shards | shard | stash | nodeid` rows. The `stash` column is the exact Jenkins stash name.

For an archived Jenkins build, download or open `reports/shard_map.txt` and search for the nodeid or any useful substring. Each row is grep-friendly:

```text
nodeid=<nodeid> stage=<stage> shard=<i>/<n> stash=<stash> group=<group> weight_s=<seconds> source=<known|fallback|pinned|round_robin>
```

### Add a new test?

Decorate it with the existing marker, for example `@pytest.mark.fpgadataflow`, `@pytest.mark.end2end`, or `@pytest.mark.bnn_u250`. The next CI run picks it up automatically.

### Add a new BNN parameter value?

Edit the `_BNN_WBITS`, `_BNN_ABITS`, and `_BNN_TOPOLOGY` constants in `tests/end2end/test_end2end_bnn_pynq.py`. Nothing else is needed.

### Add a new HW test type?

Adding a new `hwTestType` (today `bnn_build_sanity` or `bnn_build_full`) is a one-line change in [`ci_sharding.py`](./ci_sharding.py): add `STAGES` rows whose `zipArtifacts.hwTestType` is the new name and an entry in `HW_TEST_TYPE_LABELS` mapping that name to the human-readable label that should appear in the Jenkins UI (e.g. `"bnn_build_robust": "Robust"`). The `hw-test-types-json` and `hw-test-type-labels-json` subcommands pick the new entry up automatically in first-appearance order (so place smoke/sanity types before longer test types), and `validate_config()` rejects an `hwTestType` declared in `STAGES` without a matching label.

### Add a new BNN board?

1. Add the marker `bnn_<board>` to `setup.cfg` under `[tool:pytest]`.
2. In [`ci_sharding.py`](./ci_sharding.py), add a `BOARDS` entry (in the desired test-parametrisation position; key order is load-bearing) with `agentLabel`, `credentialsId`, `restartPrep`, `setupScript`, `marker`, and a `STAGES` row that references the board in its `zipArtifacts.boards`. `TEST_BOARDS` is derived from `BOARDS` and is consumed by `tests/end2end/test_end2end_bnn_pynq.py` automatically.
3. Add a `_BNN_MARKER_BY_BOARD` line in `tests/end2end/test_end2end_bnn_pynq.py`. `validate_config()` cross-checks the marker tables and sanity-checks each row's marker/shards/workers/distMode on every CLI invocation. `Jenkinsfile_HW` derives `HW_SHARDS` from `BOARDS` via `python3 docker/jenkins/ci_sharding.py hw-shards-json` and the HW test type list from `hw-test-types-json`, so no separate edit is needed there.

### Add a new CI param?

`ci_sharding.STAGES` rows carry a `param` field that maps onto the `STAGES` Jenkins choice. To add a new family (say `quantization`):

1. Add `STAGES` rows with `"param": "quantization"`.
2. Run `python3 docker/jenkins/ci_sharding.py stage-choices-json` and mirror the generated list in [`Jenkinsfile`](./Jenkinsfile)'s declarative `choice` block. The `test_jenkinsfile_stage_choices_match_python_source` test catches drift.
3. Add a row to the `STAGES` table further down in this README. The `test_readme_stages_table_matches_python_source` test catches drift.

`enabled_params_for_choice` is dynamic and picks up the new name automatically.

### Trigger a build for a contribution PR?

`STAGES` is the only knob most contributors touch. The short summary:

| `STAGES` value | Rows that run | Use when | Needs `FINN_CI_NFS_ROOT`? |
|----------------|---------------|----------|---------------------------|
| `sanity` (default) | Sanity rows only | Per-PR quick check | Recommended (publishes `bnn_build_sanity` zips for HW handoff) |
| `full` | Every CI row | Nightly / pre-merge full matrix | Yes (otherwise no handoff and no timing master update) |
| `fpgadataflow` | fpgadataflow row(s) only | Pure build-side debug, no HW handoff produced | No |
| `end2end` | end2end + BNN rows only | Debugging just the end2end family | Recommended (BNN rows publish `bnn_build_full` zips) |

If `FINN_CI_NFS_ROOT` is unset, any row that would publish board artifacts no-ops at handoff time and the aggregate stage marks the build UNSTABLE with a per-board summary.

Other narrower knobs: `local_setup` is an orthogonal opt-in for the non-Docker Vivado setup test. `STAGE_FILTER` is a substring filter for debugging a single shard within whichever rows `STAGES` selected. The HW job exposes `build_job_name`, `build_dir`, and `allow_fallback` for handoff overrides.

### Debug one stage without running the whole pipeline?

Trigger a build with the matching `STAGES` value plus a `STAGE_FILTER` substring. `STAGE_FILTER` is substring-matched against the shard's display name (`<row.stage>` for `shards=1` rows, `<row.stage> (<i>/<N>)` for sharded rows). Example: to rerun only the U250 BNN shards, set `STAGES=end2end` and `STAGE_FILTER=BNN U250`.

### Pin a flaky test to a specific shard?

`@pytest.mark.shard(N)` pins a test and its `xdist_group` siblings to shard N. If the pinned test shares an `xdist_group` with siblings, pin every member explicitly to avoid splitting the chain.

### Preview a shard split before pushing?

```bash
pytest --collect-only --num-shards 4 --dry-run-shards -m '<marker>'
```

This prints `shard | items | groups | weight_s | sample_group` and exits. Set `FINN_CI_TIMINGS_FILE=<path>` to preview with a synthetic or archived timing state.

### Inspect timing state?

Open `reports/ci_timings_master.json` from any archived build. The `last_update` field shows the build's accepted/rejected/force_accepted/anomaly counts and whether it was a persistent update. Manual updates are not supported. Run a successful `STAGES=full` build to refresh the shared master.

`python3 docker/jenkins/ci_sharding.py summarize path/to/reports/` prints per-shard wall-clock outliers from archived `<stash>.timings.json` files, useful when investigating one slow build.

## Artefacts

- `reports/*.xml`, `reports/*.html` from pytest and `pytest_html_merger`.
- `reports/<stash>.timings.json` per shard.
- `reports/<stash>.shardmap.txt` and `reports/<stash>.shardmap.json` per shard.
- `reports/shard_map.txt` and `reports/shard_map.json` merged across all shards.
- `reports/ci_timings_master.json` archived timing preview from this build. Its `last_update` field records the accepted/rejected/force_accepted/anomaly counts and whether the shared master was updated.
- `reports/<stash>.empty-shard` per shard that collected zero items. Useful for distinguishing "shard had no work" from "shard crashed".
- `coverage_combined/` one merged HTML report across all rows with `coverage: true`. Per-shard pytest runs write raw `.coverage` data files (one per shard, named via `COVERAGE_FILE=<stash>.coverage`), `aggregateReports` runs `coverage combine` and `coverage html` on the union, and the merged result is archived. Skipped silently when no row opted in.
- `${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD_NUMBER>/zips/<hwTestType>/<board>.zip` per row with a `zipArtifacts` entry. `aggregateReports()` runs `assertZipArtifactsEmitted()` which marks the build UNSTABLE (non-fatal) when an active row declared `zipArtifacts` but no `.READY` was written.
- `${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD_NUMBER>/zips/<hwTestType>/<board>.zip.READY` per-board handshake marker, touched only after the zip is in place. Publishing is idempotent for same-build retries.
- `${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD_NUMBER>/BUILD_INFO.txt` for human traceability.

## HW pipeline

[`Jenkinsfile_HW`](./Jenkinsfile_HW) runs the board-tethered tests. It is structured around `HW_SHARDS` (derived from `BOARDS` in `ci_sharding.py`) and `HW_TEST_TYPES` (discovered from `STAGES[*].zipArtifacts.hwTestType`, with display names from `HW_TEST_TYPE_LABELS`). Offline boards are gated by `isNodeOnline(label)` so the pipeline never enters `node(offlineLabel)`, and an offline board with a READY zip marks the HW build unstable instead of silently passing. Board+testType pairs with no READY zip resolved by `resolveBuildBoardZipPaths()` are skipped, so HW jobs run cleanly against partial build runs.

The Collect stage populates a `BUILD_BOARD_ZIP_PATHS` map keyed by `<testType>_<board>`. Every later step is a pure map lookup, no filesystem stats run outside the Collect node.

Build resolution is automatic via `build_job_name` (default `finn`). See "Build-to-HW zip handoff" above. `build_dir` is an optional explicit override (full path to a single build directory) for off-Jenkins recovery or pinning a specific build.

Board zips are retrieved by direct filesystem read from each board's resolved build directory on the `finn-build` aggregator agent, then `stash`/`unstash`'d to the board agents. Boards whose zip has no `.READY` sibling are skipped from the per-board stash and per-shard branch maps. If a whole HW stage has no READY zips, the stage marks the build unstable with a clear message instead of calling `parallel` with an empty branch map.

Shared Groovy helpers live in [`_common.groovy`](./_common.groovy). Each Jenkinsfile loads it once via `load 'docker/jenkins/_common.groovy'` and exposes thin top-level wrappers. `safeStash*` and `cleanPreviousBuildFiles*` split into build-pipeline and HW forms because the two pipelines genuinely disagree on what to include and how to clean. HW takes an explicit `fileBase` (its reports use a different basename from its stash name) and uses sudo when board credentials are bound. The build pipeline stashes the full sidecar set and pre-creates the dir as the unprivileged user so docker `-v` does not bind it as root. `aggregateReports` stays inline in each Jenkinsfile because the build-pipeline form does many more things than the HW form.
