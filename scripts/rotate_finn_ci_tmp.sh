#!/usr/bin/env bash
# Prune old per-build FINN CI trees under:
#   <agent_nfs_root>/workspace/tmp/ci_runs/<job_key>/<BUILD_NUMBER>/
# Invoked from docker/jenkins/Jenkinsfile (Validate): keeps the largest N
# numeric build directories, always keeps the current build, and removes any
# directory whose mtime is older than max_age days (except current). See
# docker/jenkins/README.md.
#
# Usage:
#   rotate_finn_ci_tmp.sh <agent_nfs_root> <job_key> <current_build> <retain_n> <max_age_days> [--dry-run]
#
set -euo pipefail

agent_root=${1:?agent_nfs_root}
job_key=${2:?job_key}
current=${3:?current_build}
retain_n=${4:?retain_n}
max_age_days=${5:?max_age_days}
shift 5 || true
dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
  dry_run=1
fi

if ! [ "$retain_n" -ge 1 ] 2>/dev/null; then
  echo "rotate_finn_ci_tmp: retain_n must be >= 1 (got ${retain_n})" >&2
  exit 1
fi

base="${agent_root}/workspace/tmp/ci_runs/${job_key}"
mkdir -p "${base}"

# Numeric top-level build dirs only, sorted
nums=()
while IFS= read -r line; do
  [ -n "$line" ] && nums+=("$line")
done < <(find "${base}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n)

if [ ${#nums[@]} -eq 0 ]; then
  echo "rotate_finn_ci_tmp: no build dirs under ${base}"
  exit 0
fi

# Last retain_n entries = largest N build numbers
c=${#nums[@]}
if [ "$c" -le "$retain_n" ]; then
  protected=("${nums[@]}")
else
  start=$((c - retain_n))
  protected=()
  for ((i = start; i < c; i++)); do
    protected+=("${nums[i]}")
  done
fi
# Always retain current (may not exist on disk yet, or may be below top N)
found=0
for p in "${protected[@]}"; do
  if [ "$p" = "$current" ]; then
    found=1
    break
  fi
done
if [ "$found" -eq 0 ]; then
  protected+=("$current")
fi

should_delete() {
  local d=$1
  local path="${base}/${d}"
  if [ "$d" = "$current" ]; then
    return 1
  fi
  if [ "$max_age_days" -gt 0 ] && [ -d "$path" ]; then
    if find "$path" -maxdepth 0 -mtime +"$max_age_days" 2>/dev/null | grep -q .; then
      return 0
    fi
  fi
  for p in "${protected[@]}"; do
    if [ "$d" = "$p" ]; then
      return 1
    fi
  done
  return 0
}

for n in "${nums[@]}"; do
  if should_delete "$n"; then
    if [ "$dry_run" -eq 1 ]; then
      echo "rotate_finn_ci_tmp: would delete ${base}/${n}"
    else
      echo "rotate_finn_ci_tmp: deleting ${base}/${n}"
      rm -rf "${base:?}/${n}"
    fi
  fi
done

echo "rotate_finn_ci_tmp: done (base=${base} current=${current} retain_n=${retain_n} max_age_days=${max_age_days} dry_run=${dry_run})"
