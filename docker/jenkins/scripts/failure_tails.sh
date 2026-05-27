#!/bin/bash
# failure_tails.sh <build_dir> <stash> <tail_lines>
#
# Tail every tool log with an ERROR: marker so the Jenkins console shows the
# actual error without downloading the artifact tarball.
set +e

if [ "$#" -ne 3 ]; then
  echo "Usage: failure_tails.sh <build_dir> <stash> <tail_lines>" >&2
  exit 2
fi

bd=$1
stash=$2
tail_lines=$3
tag="[failure-tails ${stash}]"

if [ ! -d "$bd" ]; then
  echo "$tag no build dir"
  exit 0
fi

mapfile -t failed < <(find "$bd" \( \
    -name vitis_hls.log -o \
    -name build_dataflow.log -o \
    -path '*/project_*/sol1/impl/ip/vivado.log' -o \
    -path '*/finn_zynqbuild_*/vivado.log' -o \
    -path '*/vitis_proj/_x/logs/*.log' \
  \) -exec grep -l 'ERROR:' {} + 2>/dev/null)

if [ "${#failed[@]}" = "0" ]; then
  echo "$tag no logs with ERROR: markers found"
  exit 0
fi

echo "$tag ${#failed[@]} log file(s) with ERROR: markers"
for f in "${failed[@]}"; do
  rel="${f#"$bd"/}"
  echo ""
  echo "=== FAIL: $rel (tail ${tail_lines}) ==="
  tail -n "$tail_lines" "$f"
done
