#!/usr/bin/env bash
# ctf-rs acceptance run (Linux/WSL, and macOS for local smoke runs).
#
# Reads scripts/acceptance-targets.tsv (checked by scripts/check-targets.sh)
# for the set of gating cargo test targets, builds every needed test binary
# once, then runs each one, recording a per-target RUN_EXIT line and log so
# a failure partway through does not void the evidence of everything that
# ran after it (audit findings E1, E2, E4, E5). mpi targets run the built
# executable directly under `mpirun`, bypassing Cargo's per-host-triple
# runner variable, so this script does not hard-code a target triple and
# runs unchanged on Linux, WSL, and macOS. lib and local targets run through
# `cargo test --no-fail-fast` instead, since Cargo already resolves those
# for the host without a runner override.
#
# Env overrides:
#   CTF_ACCEPTANCE_LOG_DIR   log directory (default: a timestamped dir under
#                            $HOME/.cache/ctf-rs-acceptance)
#   CTF_ACCEPTANCE_ONLY      space-separated target names to restrict to
#                            (smoke runs); default is every manifest target
#   CTF_ACCEPTANCE_RANKS     space-separated rank counts to use for every
#                            mpi target, overriding the manifest's per-target
#                            ranks column (default: the manifest's ranks)
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/ctf-rs-target}"
export OPENBLAS_NUM_THREADS=1

manifest="scripts/acceptance-targets.tsv"
bash scripts/check-targets.sh

mpirun_bin="$(command -v mpirun || true)"
if [[ -z "$mpirun_bin" ]]; then
  echo "acceptance-wsl: mpirun not found on PATH" >&2
  exit 1
fi

timeout_bin="$(command -v timeout || command -v gtimeout || true)"
if [[ -z "$timeout_bin" ]]; then
  echo "acceptance-wsl: warning: no 'timeout'/'gtimeout' on PATH, running mpi targets without a wall-clock cap" >&2
fi

logdir="${CTF_ACCEPTANCE_LOG_DIR:-$HOME/.cache/ctf-rs-acceptance/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$logdir"
echo "acceptance-wsl: logs in $logdir"

only_filter="${CTF_ACCEPTANCE_ONLY:-}"
ranks_override="${CTF_ACCEPTANCE_RANKS:-}"

is_selected() {
  local target="$1"
  [[ -z "$only_filter" ]] && return 0
  local name
  for name in $only_filter; do
    [[ "$name" == "$target" ]] && return 0
  done
  return 1
}

# macOS ships bash 3.2, which raises "unbound variable" under `set -u` for
# `"${array[@]}"` (but not `"${!array[@]}"`) when the array is empty; every
# `[@]` expansion below of an array that CTF_ACCEPTANCE_ONLY can leave empty
# uses the `${array[@]+"${array[@]}"}` guard so the script stays portable.
mpi_targets=()
mpi_ranks=()
lib_filters=()
local_targets=()

while IFS=$'\t' read -r target class ranks sets note; do
  [[ "$target" == "target" ]] && continue
  [[ -z "$target" ]] && continue
  is_selected "$target" || continue
  case "$class" in
    mpi) mpi_targets+=("$target"); mpi_ranks+=("$ranks") ;;
    lib) lib_filters+=("$target") ;;
    local) local_targets+=("$target") ;;
    excluded) : ;;
    *)
      echo "acceptance-wsl: unknown class '$class' for target '$target' in $manifest" >&2
      exit 1
      ;;
  esac
done < "$manifest"

if [[ -n "$only_filter" ]] && (( ${#mpi_targets[@]} + ${#lib_filters[@]} + ${#local_targets[@]} == 0 )); then
  echo "acceptance-wsl: CTF_ACCEPTANCE_ONLY matched no gating target in $manifest: $only_filter" >&2
  exit 1
fi

# Build every selected mpi/local test binary once, plus the lib test binary
# if any lib filter is selected, and capture cargo's JSON build artifacts so
# each mpi target's executable path can be resolved without a runner
# env var. Building is still setup: a build failure aborts under set -e.
build_args=()
for t in "${mpi_targets[@]+"${mpi_targets[@]}"}" "${local_targets[@]+"${local_targets[@]}"}"; do
  build_args+=(--test "$t")
done
if (( ${#lib_filters[@]} > 0 )); then
  build_args+=(--lib)
fi

build_json="$logdir/build.json"
build_log="$logdir/build.stderr.log"
if (( ${#build_args[@]} > 0 )); then
  cargo test --no-run --message-format=json "${build_args[@]}" \
    >"$build_json" 2>"$build_log"
fi

# Portable stand-in for an associative array: macOS ships bash 3.2, which
# has no `declare -A`, so the executable map is two parallel arrays plus a
# linear-scan lookup function instead.
exe_names=()
exe_paths=()
if [[ -s "$build_json" ]]; then
  while IFS=$'\t' read -r name exe; do
    exe_names+=("$name")
    exe_paths+=("$exe")
  done < <(python3 -c '
import json, sys
with open(sys.argv[1]) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        profile = msg.get("profile") or {}
        if msg.get("reason") == "compiler-artifact" and profile.get("test") is True and msg.get("executable"):
            print(msg["target"]["name"] + "\t" + msg["executable"])
' "$build_json")
fi

exe_for() {
  local want="$1" i
  for i in "${!exe_names[@]}"; do
    if [[ "${exe_names[$i]}" == "$want" ]]; then
      printf '%s' "${exe_paths[$i]}"
      return 0
    fi
  done
  return 1
}

for t in "${mpi_targets[@]+"${mpi_targets[@]}"}"; do
  if ! exe_for "$t" >/dev/null; then
    echo "acceptance-wsl: no built executable for mpi target '$t' (see $build_log)" >&2
    exit 1
  fi
done

# Setup is done; from here a failing target must not stop the run (E2).
set +e

pass_count=0
fail_count=0
failing=()

record() {
  local label="$1" code="$2"
  if [[ "$code" -eq 0 ]]; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
    failing+=("$label exit=$code")
  fi
}

for idx in "${!mpi_targets[@]}"; do
  target="${mpi_targets[$idx]}"
  manifest_ranks="${mpi_ranks[$idx]}"
  ranks_list="${ranks_override:-$manifest_ranks}"
  exe="$(exe_for "$target")"
  for ranks in $ranks_list; do
    log="$logdir/${target}-${ranks}.log"
    if [[ -n "$timeout_bin" ]]; then
      "$timeout_bin" 600 "$mpirun_bin" --oversubscribe -n "$ranks" "$exe" >"$log" 2>&1
    else
      "$mpirun_bin" --oversubscribe -n "$ranks" "$exe" >"$log" 2>&1
    fi
    code=$?
    echo "RUN_EXIT $target ranks=$ranks exit=$code"
    record "$target ranks=$ranks" "$code"
  done
done

for filter in "${lib_filters[@]+"${lib_filters[@]}"}"; do
  log="$logdir/lib-${filter}.log"
  cargo test --lib "$filter" --no-fail-fast -- --nocapture >"$log" 2>&1
  code=$?
  echo "RUN_EXIT $filter ranks=- exit=$code"
  record "$filter" "$code"
done

for target in "${local_targets[@]+"${local_targets[@]}"}"; do
  log="$logdir/${target}.log"
  cargo test --test "$target" --no-fail-fast -- --nocapture >"$log" 2>&1
  code=$?
  echo "RUN_EXIT $target ranks=- exit=$code"
  record "$target" "$code"
done

total=$((pass_count + fail_count))
echo "acceptance-wsl: summary $pass_count/$total passed"
if (( fail_count > 0 )); then
  echo "acceptance-wsl: failing ($fail_count):"
  for f in "${failing[@]+"${failing[@]}"}"; do
    echo "  $f"
  done
  exit 1
fi
exit 0
