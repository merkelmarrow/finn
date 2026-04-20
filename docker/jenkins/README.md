# FINN Jenkins CI — stage lookup

This file is the **human-readable index** of what each parallel stage in
[`Jenkinsfile`](./Jenkinsfile) runs. If you want to know which stage exercises
a given test, marker, or board, start here.

## Quick answers

| Question                                                 | Where to look                                                                                |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Which stage runs test `tests/foo/test_bar.py::baz`?       | `./scripts/list_ci_stages.py` and grep, or `--collect 'Stage Name'`                          |
| Why is this stage so slow?                               | Row `workers:` column below; tune in `BNN_SUB_STAGES` / `END2END_SHARDS` in `Jenkinsfile`    |
| How do I move a fpgadataflow test between shards?         | Add/remove `pytestmark = [pytest.mark.fpgadataflow_slow]` in the test file; no Jenkinsfile edit |
| How do I add a new BNN sub-stage?                        | Mark the target tests, register the marker in `setup.cfg`, add a row to `BNN_SUB_STAGES`    |
| What does this stage actually execute?                   | Jenkins console log prints `runPytest[<stash>]: python -m pytest …` before every invocation |

## Stage → pytest / agent / stash table

All stages run on a `finn-build` agent. The `Build Docker Image` and
`Check Stage Results` stages are sequential; everything else runs in the
`Run Tests` parallel block, gated by the `sanity`/`fpgadataflow`/`end2end`
job parameters.

| Stage                       | Param         | Marker expression                                                         | Stash                      | Extra artefacts                              |
| --------------------------- | ------------- | ------------------------------------------------------------------------- | -------------------------- | -------------------------------------------- |
| Build Docker Image          | (always)      | —                                                                         | —                          | `finn-docker-image.tar.gz` (shared dir)      |
| Sanity - Build Hardware     | sanity\*      | `sanity_bnn`                                                              | `bnn_build_sanity`         | `{U250,Pynq-Z1,ZCU104,KV260_SOM}.zip`        |
| Sanity - Unit Tests         | sanity        | `util or brevitas_export or streamline or transform or notebooks`         | `sanity_ut`                | `coverage_sanity_ut/`                        |
| fpgadataflow - shard A      | fpgadataflow  | `fpgadataflow and fpgadataflow_slow`                                      | `fpgadataflow_a`           | `coverage_fpgadataflow_a/`                   |
| fpgadataflow - shard B      | fpgadataflow  | `fpgadataflow and not fpgadataflow_slow`                                  | `fpgadataflow_b`           | `coverage_fpgadataflow_b/`                   |
| BNN \<Board\> - \<label\>   | end2end       | `bnn_<board> and <scenario markers>` (see `BNN_SUB_STAGES`)               | `bnn_<Board>_<label>`      | `<Board>.zip` when label == `cnv-w2a2`       |
| End2end - mobilenet         | end2end       | `end2end` on `tests/end2end/test_end2end_mobilenet_v1.py`                 | `end2end_mobilenet`        | —                                            |
| End2end - rest              | end2end       | `end2end` on the remaining four `tests/end2end/*.py` files                | `end2end_rest`             | —                                            |
| Check Stage Results         | (always)      | — (`junit` + `pytest_html_merger`)                                        | —                          | `reports/*.xml`, `reports/*.html`            |

\* Sanity-Build-Hardware is **skipped when `end2end` is also set** — the BNN
sub-stages already rebuild the same four sanity scenarios.

## Adding/removing tests from the heavy fpgadataflow shard

The heavy shard is defined by the `fpgadataflow_slow` marker. To rebalance:

```
./scripts/balance_fpgadataflow_shards.py reports/fpgadataflow_*.xml
```

prints `ADD` / `REMOVE` recommendations for the module-level `pytestmark =
[pytest.mark.fpgadataflow_slow]` in each test file. Apply those edits in the
test files; no Jenkinsfile change is needed.

## Adding a new BNN scenario

1. Pick or add a scenario marker (e.g. `bnn_cnv_w4a4`) in
   `tests/end2end/test_end2end_bnn_pynq.py::pytest_generate_tests`.
2. Register it under `[tool:pytest]` `markers` in `setup.cfg`.
3. Add a row to `BNN_SUB_STAGES` with `marker: 'bnn_<board> and bnn_cnv_w4a4'`
   and add its stash to `expectedStashes()` automatically via the loop there.

## See also

- `scripts/list_ci_stages.py` — source-of-truth Python extractor
- `knowledge/platform/jenkins_ci.md` — agent / credential / bug history
- `knowledge/analyses/finn_ci_12h_15exec_parallelisation.md` — wall-clock budget
