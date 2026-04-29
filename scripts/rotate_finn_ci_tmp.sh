#!/usr/bin/env bash
# Prune old per-build FINN CI trees under every per-agent subtree of the
# shared NFS base:
#   <nfs_root_base>/<NODE_NAME>/workspace/tmp/ci_runs/<job_key>/<BUILD_NUMBER>/
#
# Invoked once per build from docker/jenkins/Jenkinsfile (Validate). Walks
# every <nfs_root_base>/*/workspace/tmp/ci_runs/<job_key>/ on disk so the
# multi-agent fan-out (xircseeng01 + xirdcglab*) is rotated by a single
# call regardless of which agent picked up Validate.
#
# Per-tree policy:
#   - Always keep <current_build>.
#   - Keep the largest <retain_n> numeric build dirs.
#   - Of the remaining (older) dirs, delete any whose mtime exceeds
#     <max_age_days>. Top-N protection wins over mtime, so an idle job
#     does not lose its recent history.
#
# Usage:
#   rotate_finn_ci_tmp.sh <nfs_root_base> <job_key> <current_build> <retain_n> <max_age_days> [--dry-run]
#
set -euo pipefail

nfs_base=${1:?nfs_root_base}
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

if [ ! -d "$nfs_base" ]; then
  echo "rotate_finn_ci_tmp: nfs_root_base ${nfs_base} not present, skipping"
  exit 0
fi

rotate_one_tree() {
  local base=$1
  if [ ! -d "$base" ]; then
    return 0
  fi

  local nums=()
  while IFS= read -r line; do
    [ -n "$line" ] && nums+=("$line")
  done < <(find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
             | grep -E '^[0-9]+$' | sort -n)

  if [ ${#nums[@]} -eq 0 ]; then
    echo "rotate_finn_ci_tmp: ${base}: no build dirs"
    return 0
  fi

  local c=${#nums[@]}
  local protected=()
  if [ "$c" -le "$retain_n" ]; then
    protected=("${nums[@]}")
  else
    local start=$((c - retain_n))
    local i
    for ((i = start; i < c; i++)); do
      protected+=("${nums[i]}")
    done
  fi

  # Always keep current build, even if it is below the top-N cut or has
  # not yet written its dir on this agent (no-op append in that case).
  local found=0 p
  for p in "${protected[@]}"; do
    if [ "$p" = "$current" ]; then
      found=1
      break
    fi
  done
  if [ "$found" -eq 0 ]; then
    protected+=("$current")
  fi

  local n path is_protected
  for n in "${nums[@]}"; do
    is_protected=0
    for p in "${protected[@]}"; do
      if [ "$n" = "$p" ]; then
        is_protected=1
        break
      fi
    done
    if [ "$is_protected" -eq 1 ]; then
      continue
    fi

    path="${base}/${n}"
    # mtime gate: only delete if older than max_age_days. Setting
    # max_age_days=0 disables the gate and prunes every non-protected
    # entry on every run.
    if [ "$max_age_days" -gt 0 ]; then
      if ! find "$path" -maxdepth 0 -mtime +"$max_age_days" 2>/dev/null | grep -q .; then
        continue
      fi
    fi

    if [ "$dry_run" -eq 1 ]; then
      echo "rotate_finn_ci_tmp: would delete ${path}"
    else
      echo "rotate_finn_ci_tmp: deleting ${path}"
      rm -rf "${base:?}/${n}"
    fi
  done
}

# Per-agent subtree layout matches withAgentNfsEnv() in the Jenkinsfile.
# We walk one level under nfs_base to discover every agent that has ever
# run a finn build, even if the agent is currently offline.
trees_visited=0
for agent_root in "${nfs_base}"/*/; do
  [ -d "$agent_root" ] || continue
  base="${agent_root%/}/workspace/tmp/ci_runs/${job_key}"
  if [ -d "$base" ]; then
    rotate_one_tree "$base"
    trees_visited=$((trees_visited + 1))
  fi
done

echo "rotate_finn_ci_tmp: done (nfs_base=${nfs_base} job_key=${job_key} current=${current} retain_n=${retain_n} max_age_days=${max_age_days} dry_run=${dry_run} trees=${trees_visited})"
