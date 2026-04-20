#!/usr/bin/env python3
"""Expand every Jenkins parallel stage to the pytest command it will run.

Regex-parses docker/jenkins/Jenkinsfile (NO Groovy/Jenkins imports) so a dev
can answer "which stage runs this test?" without a CI round-trip:

    ./scripts/list_ci_stages.py           # stage name | stash | pytest cmd
    ./scripts/list_ci_stages.py | grep fpgadataflow_slow

Only dynamic tables (BNN_SUB_STAGES, END2END_SHARDS, fpgadataflow shards) are
expanded; fixed sanity stages are listed verbatim for completeness.
"""
import re
import sys
from pathlib import Path

JF = Path(__file__).resolve().parents[1] / "docker/jenkins/Jenkinsfile"

BNN = re.compile(
    r"\[board:\s*'([^']+)',\s*label:\s*'([^']+)',\s*marker:\s*'([^']+)',\s*workers:\s*(\d+)\]"
)
E2E = re.compile(
    r"\[label:\s*'([^']+)',\s*workers:\s*(\d+),\s*files:\s*\[(.*?)\]\]", re.DOTALL
)


def main():
    text = JF.read_text()
    rows = [
        ("Sanity - Build Hardware", "bnn_build_sanity", "pytest -m 'sanity_bnn'"),
        ("Sanity - Unit Tests", "sanity_ut",
         "pytest -m 'util or brevitas_export or streamline or transform or notebooks' -n $WORKERS"),
        ("fpgadataflow - shard A", "fpgadataflow_a",
         "pytest -m 'fpgadataflow and fpgadataflow_slow' -n $WORKERS"),
        ("fpgadataflow - shard B", "fpgadataflow_b",
         "pytest -m 'fpgadataflow and not fpgadataflow_slow' -n $WORKERS"),
    ]
    for board, label, marker, workers in BNN.findall(text):
        stash = ("bnn_%s_%s" % (board, label)).replace("-", "_")
        rows.append(("BNN %s - %s" % (board, label), stash,
                     "pytest -m '%s' -n %s --dist loadgroup" % (marker, workers)))
    for label, workers, files_blob in E2E.findall(text):
        files = " ".join(re.findall(r"'([^']+)'", files_blob))
        rows.append(("End2end - %s" % label, "end2end_%s" % label,
                     "pytest -m 'end2end' %s -n %s --dist loadgroup" % (files, workers)))
    nw = max(len(r[0]) for r in rows)
    sw = max(len(r[1]) for r in rows)
    for name, stash, cmd in rows:
        print("%-*s  %-*s  %s" % (nw, name, sw, stash, cmd))


if __name__ == "__main__":
    sys.exit(main() or 0)
