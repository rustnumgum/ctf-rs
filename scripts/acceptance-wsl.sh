#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="$HOME/.cache/ctf-rs-target"
export OPENBLAS_NUM_THREADS=1

# Run in Linux, not as an interpolated PowerShell command string.
# Each required rank configuration runs once; failed commands stop the set.
for ranks in 1 2 4; do
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="mpirun --oversubscribe -n $ranks" \
    cargo test --test foundation
done
for ranks in 1 2 4; do
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="mpirun --oversubscribe -n $ranks" \
    cargo test --test dense_views
done
cargo test --test local_linalg -- --nocapture
