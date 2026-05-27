#!/bin/bash
# archive_failure_logs.sh <build_dir> <tarball_path> [start_marker]
#
# One tarball of tool logs per failed shard. Excludes HLS compiler
# intermediates. LSF staging logs live outside the build dir
# (FINN_LSF_NFS_STAGING) so they need a separate find rooted there, scoped to
# files touched since the fixed shard-start marker.
# Best-effort: missing build dir is benign, the caller passes
# allowEmptyArchive=true to archiveArtifacts.
set +e

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: archive_failure_logs.sh <build_dir> <tarball_path> [start_marker]" >&2
  exit 2
fi

bd=$1
tarball=$2
start_marker=${3:-}
lsf_staging="${FINN_LSF_NFS_STAGING:-}"

mkdir -p "$(dirname "$tarball")"
if [ ! -d "$bd" ]; then
  exit 0
fi

# tar reads the file list from stdin, and the find on the left of the pipe
# runs in its own subshell, so emit absolute paths so tar (right of the pipe)
# can stat them from its own cwd. The pre-extraction heredoc emitted relative
# "./..." paths after a `cd`, which tar then could not find, and the resulting
# tarballs were silently empty (allowEmptyArchive=true masked it).
abs_bd=$(cd "$bd" && pwd)
newer_ref=$bd
if [ -n "$start_marker" ] && [ -e "$start_marker" ]; then
  newer_ref=$start_marker
fi
{
  find "$abs_bd" -type f \( \
      -name vitis_hls.log -o \
      -name build_dataflow.log -o \
      -path '*/project_*/sol1/impl/ip/vivado.log' -o \
      -path '*/finn_zynqbuild_*/vivado.log' -o \
      -path '*/vitis_proj/_x/logs/*.log' \
    \) -print0 2>/dev/null
  if [ -d "$lsf_staging" ]; then
    find "$lsf_staging" -mindepth 2 -maxdepth 3 -type f -newer "$newer_ref" \( \
        -name 'lsf.stdout' -o \
        -name 'lsf.stderr' -o \
        -name 'remote_runner.sh' \
      \) -print0 2>/dev/null
  fi
} | tar --null --create --gzip --file "$tarball" --files-from - 2>/dev/null || true
