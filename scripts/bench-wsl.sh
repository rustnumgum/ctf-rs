#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="$HOME/.cache/ctf-rs-target"
export OPENBLAS_NUM_THREADS=1
cargo build --release --example mpi_gemm_bench
for ranks in 1 2 4; do
  report="$HOME/.cache/ctf-rs-bench/ranks-$ranks"
  mkdir -p "$report"
  mpirun --oversubscribe -n "$ranks" bash scripts/time-rank.sh "$report" \
    "$CARGO_TARGET_DIR/release/examples/mpi_gemm_bench" | tee "$report/elapsed.txt"
  grep 'Maximum resident set size' "$report"/rank-*.time
done
