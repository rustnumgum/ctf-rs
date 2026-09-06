#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="$HOME/.cache/ctf-rs-target"
export OPENBLAS_NUM_THREADS=1

# Run in Linux, not as an interpolated PowerShell command string.
# Each required rank configuration runs once; failed commands stop the set.
mpi_tests=(foundation dense_views replicated_sum tensor_sum custom_reduce algebra_sum
  complex_scalar sum_remap replicated_contraction ctr_2d tensor_gemm
  algebra_contraction tensor_contract contract_remap dense_semantics upstream_dense subcomm_dense plan_cache model_training selector)
args=()
for test in "${mpi_tests[@]}"; do args+=(--test "$test"); done
for ranks in 1 2 4; do
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="mpirun --oversubscribe -n $ranks" \
    cargo test "${args[@]}"
done
cargo test --test local_linalg --test topology_candidates --test node_aware --test map_tensor \
  --test sequential_sum --test virtual_sum --test sequential_contraction --test folded_contraction \
  --test sparse_formats --test cost_models -- --nocapture
