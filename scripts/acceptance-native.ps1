param(
    [string]$MingwRoot = 'C:\msys64\mingw64',
    [string]$TargetDir = 'D:\ctf-rs-native-target',
    [switch]$BuildOnly
)
$ErrorActionPreference = 'Stop'
$env:PATH = "$HOME\.cargo\bin;$MingwRoot\bin;C:\Program Files\Microsoft MPI\Bin;" + $env:PATH
$env:MSMPI_INC = "$MingwRoot\include"
$env:MSMPI_LIB64 = "$MingwRoot\lib"
$env:LIBCLANG_PATH = "$MingwRoot\bin"
$env:RUSTFLAGS = "-L native=$($MingwRoot.Replace('\','/'))/lib"
$env:CARGO_TARGET_DIR = $TargetDir
$env:OPENBLAS_NUM_THREADS = '1'
if ($BuildOnly) {
    cargo test --tests --no-run
    exit $LASTEXITCODE
}

$mpiTests = @(
    'foundation','dense_views','replicated_sum','tensor_sum','custom_reduce',
    'algebra_sum','complex_scalar','sum_remap','replicated_contraction','ctr_2d',
    'tensor_gemm','algebra_contraction','tensor_contract','contract_remap',
    'dense_semantics','upstream_dense','subcomm_dense','plan_cache','model_training',
    'selector','selection_objective','tensor_blas_fold','upstream_gemm4d',
    'distributed_matrix','distributed_qr_svd','distributed_svd_paths','distributed_eigh',
    'distributed_spd','distributed_tttp','distributed_mttkrp','distributed_tensor_svd',
    'distributed_solve_factor','distributed_sparse_io','distributed_sparse_sum',
    'distributed_sparse_gemm','distributed_sparse_fold','upstream_sparse_mp3',
    'distributed_sparse_transform','distributed_dense_sparse','upstream_sparse_mp3_t',
    'distributed_sparse_diagonal','distributed_symmetric_io','distributed_symmetric_operations',
    'distributed_symmetric_repack','distributed_packed_sum','distributed_packed_contraction',
    'distributed_canonical_sum','distributed_hollow_sum','distributed_symmetric_diagonal',
    'distributed_sy_sum','distributed_sy_scalars','distributed_canonical_contraction',
    'distributed_symmetric_contraction','upstream_diag_sym','distributed_cross_diagonal',
    'upstream_diag_ctr','upstream_sy_times_ns','upstream_multi_tsr_sym','upstream_reduce_bcast',
    'distributed_sparse_multilinear','distributed_sparse_solve_factor','distributed_sparse_input_reduction',
    'distributed_sparse_general','distributed_sparse_function','distributed_sparse_gemm_function',
    'distributed_sparse_function_output','distributed_sparse_fold_function',
    'upstream_bivar_function','distributed_dense_function',
    'upstream_univar_function','upstream_endomorphism','upstream_bivar_transform',
    'upstream_endomorphism_cust','upstream_endomorphism_cust_sp','distributed_exhaustive_mapping','distributed_normal_mapping','selected_mapping','dense_search','dense_execution','dense_execution_algebra','dense_folded_execution','distributed_node_fold','dense_low_memory','typed_folded_execution','typed_matrix_factors','typed_distributed_qr','typed_distributed_svd','typed_svd_truncation','randomized_guess','distributed_random_fill'
)
$arguments = @()
foreach ($test in $mpiTests) { $arguments += @('--test', $test) }
foreach ($test in @('typed_grid_blas','typed_randomized_svd')) { $arguments += @('--test', $test) }
foreach ($ranks in 1,2,4) {
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = "mpiexec -n $ranks"
    cargo test @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Remove-Item Env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER
$localTests = @('random_generator','scalar_blas','node_peer_counts','folded_cost','partial_fold_kernel','fold_storage','partial_fold','fold_indices','fold_layout','fold_selection','mapped_cost','local_linalg','topology_candidates','node_aware','map_tensor',
    'sequential_sum','virtual_sum','sequential_contraction','folded_contraction',
    'sparse_formats','sparse_sequential','sparse_function','sparse_function_kernel','cost_models','plan_cost','grid_plan_cost','redist_cost','mapping_preflight','mapping_variants','topology_canonicalization','normal_mapping','symmetry_layout','sym_indices',
    'sym_triple','sym_operations','folding')
$arguments = @()
foreach ($test in $localTests) { $arguments += @('--test', $test) }
cargo test @arguments -- --nocapture
exit $LASTEXITCODE
