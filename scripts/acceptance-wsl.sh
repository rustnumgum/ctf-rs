#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="$HOME/.cache/ctf-rs-target"
export OPENBLAS_NUM_THREADS=1

# Run in Linux, not as an interpolated PowerShell command string.
# Each required rank configuration runs once; failed commands stop the set.
mpi_tests=(foundation dense_views replicated_sum tensor_sum custom_reduce algebra_sum
  complex_scalar sum_remap replicated_contraction ctr_2d tensor_gemm
  algebra_contraction tensor_contract contract_remap dense_semantics upstream_dense subcomm_dense plan_cache model_training selector selection_objective tensor_blas_fold upstream_gemm4d distributed_matrix distributed_qr_svd distributed_svd_paths distributed_eigh distributed_spd distributed_tttp distributed_mttkrp distributed_tensor_svd distributed_solve_factor distributed_sparse_io distributed_sparse_sum distributed_sparse_gemm distributed_sparse_fold upstream_sparse_mp3 distributed_sparse_transform distributed_dense_sparse upstream_sparse_mp3_t distributed_sparse_diagonal)
args=()
mpi_tests+=(distributed_symmetric_io)
mpi_tests+=(distributed_symmetric_operations)
mpi_tests+=(distributed_symmetric_repack)
mpi_tests+=(distributed_packed_sum)
mpi_tests+=(distributed_packed_contraction)
mpi_tests+=(distributed_canonical_sum)
mpi_tests+=(distributed_hollow_sum)
mpi_tests+=(distributed_symmetric_diagonal)
mpi_tests+=(distributed_sy_sum)
mpi_tests+=(distributed_sy_scalars)
mpi_tests+=(distributed_canonical_contraction)
mpi_tests+=(distributed_symmetric_contraction)
mpi_tests+=(upstream_diag_sym)
mpi_tests+=(distributed_cross_diagonal)
mpi_tests+=(upstream_diag_ctr upstream_sy_times_ns upstream_multi_tsr_sym)
mpi_tests+=(upstream_reduce_bcast)
mpi_tests+=(distributed_sparse_multilinear)
mpi_tests+=(distributed_sparse_solve_factor)
mpi_tests+=(distributed_sparse_input_reduction)
mpi_tests+=(distributed_sparse_general)
mpi_tests+=(distributed_sparse_function)
mpi_tests+=(distributed_sparse_gemm_function)
mpi_tests+=(distributed_sparse_function_output)
mpi_tests+=(distributed_sparse_fold_function)
mpi_tests+=(upstream_bivar_function distributed_dense_function)
mpi_tests+=(upstream_univar_function upstream_endomorphism upstream_bivar_transform)
mpi_tests+=(upstream_endomorphism_cust upstream_endomorphism_cust_sp)
for test in "${mpi_tests[@]}"; do args+=(--test "$test"); done
for ranks in 1 2 4; do
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="mpirun --oversubscribe -n $ranks" \
    cargo test "${args[@]}"
done
cargo test --test local_linalg --test topology_candidates --test node_aware --test map_tensor \
  --test sequential_sum --test virtual_sum --test sequential_contraction --test folded_contraction \
  --test sparse_formats --test sparse_sequential --test sparse_function --test sparse_function_kernel --test cost_models --test plan_cost --test grid_plan_cost --test redist_cost --test mapping_preflight --test mapping_variants --test symmetry_layout --test sym_indices --test sym_triple --test sym_operations --test folding -- --nocapture
