# FINN Jenkins CI — maintainer guide

Every parallel stage is defined by **one row** of `PARALLEL_SHARDS` at the
top of [`Jenkinsfile`](./Jenkinsfile). Actual test selection and shard
splitting are done by the standard `-m <marker>` expression plus the tiny
`--num-shards` / `--shard-id` plugin in [`tests/conftest.py`](../../tests/conftest.py).
No state, no allowlists, no rebalancing script.

## How do I …

### … add a new test?

Decorate it with the existing marker (`@pytest.mark.fpgadataflow`,
`@pytest.mark.end2end`, `@pytest.mark.bnn_u250`, …). That is the entire CI
change. The next CI run deterministically hashes the new test's `nodeid`
into one of the shards for that marker.

### … add a new BNN parameter value (e.g. a new `(wbits, abits)` combo)?

Edit the lists in `tests/end2end/test_end2end_bnn_pynq.py::pytest_generate_tests`.
Nothing else. The new parameter lands in some shard of the board's marker
automatically.

### … add a new BNN board?

1. Add the marker (`bnn_<board>`) to `setup.cfg` under `[tool:pytest]` `markers`.
2. Update `test_board_map` / `pytest_generate_tests` in the BNN test file.
3. Add one row to `PARALLEL_SHARDS` — `marker: 'bnn_<board>'`, a shard count
   and worker count, and `zipBoards: ['<Board>']` if it produces bitstreams
   for hardware validation.

### … handle a test that got much slower?

Nothing, usually — `worksteal`/`loadgroup` within a shard absorbs most
variance. If a whole marker's wall-clock grows out of budget, bump its
`shards:` count in `PARALLEL_SHARDS` (more Jenkins agents run it in
parallel, same total work).

### … find which stage runs a given test?

Run `pytest --collect-only -m '<marker>' <path/to/test.py>` locally. The
marker is what CI selects on. The Jenkins console log also echoes
`runPytest[<stash>]: python -m pytest …` at the start of every stage.

### … verify a marker still has CI coverage?

Any marker used in `PARALLEL_SHARDS` that collects **zero** tests will fail
the stage loudly — the plugin raises `UsageError`. A silent-skip is
impossible by construction.

## Stage → param mapping

Stages are gated by the job parameters `sanity`, `fpgadataflow`, `end2end`.
The `Sanity - Build Hardware` row is suppressed when `end2end=true` because
the BNN rows rebuild the same scenarios.

## Artefacts

- `reports/*.xml`, `reports/*.html` (merged via `pytest_html_merger`)
- `coverage_<stash>/` per row with `coverage: true`
- `<Board>.zip` per row with `zipBoards: [...]` — only emitted by the shard
  that happens to have run the tests producing `hw_deployment_*` output.
