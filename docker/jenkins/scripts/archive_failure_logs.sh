#!/bin/bash
# archive_failure_logs.sh <build_dir> <tarball_path> [start_marker]
#
# One tarball of tool logs per failed shard. LSF staging logs live outside
# the build dir under FINN_LSF_NFS_STAGING; they are scoped to files newer
# than the start_marker if provided.
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

# Use absolute paths so tar can stat them from its own cwd.
abs_bd=$(cd "$bd" && pwd)
newer_ref=$bd
if [ -n "$start_marker" ] && [ -e "$start_marker" ]; then
  newer_ref=$start_marker
fi

# Collect to a temp file so we can both count and tar, and so a healthy
# "no candidates" run does not produce an empty tarball indistinguishable
# from a tar failure.
list=$(mktemp)
trap 'rm -f "$list"' EXIT
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
} > "$list"

n=$(tr -cd '\0' < "$list" | wc -c)
echo "[archive-failure-logs] ${n} candidate file(s) for ${tarball}"
if [ "$n" = "0" ]; then
  exit 0
fi
tar --null --create --gzip --file "$tarball" --files-from "$list" 2>/dev/null || true
