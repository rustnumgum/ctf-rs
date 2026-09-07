param(
    [string]$MingwRoot = 'C:\msys64\mingw64',
    [string]$TargetDir = 'D:\ctf-rs-native-target',
    [switch]$BuildOnly,
    [switch]$D6Only
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
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    cargo build --examples
    exit $LASTEXITCODE
}

if ($D6Only) {
    $denseTests = @(
        'upstream_scalar','upstream_diag_sym','upstream_weigh4d','upstream_dft',
        'upstream_readwrite','upstream_readall','distributed_symmetric_repack',
        'upstream_permute_multiworld','upstream_reduce_bcast','upstream_subworld_gemm',
        'upstream_gemm4d','upstream_sy_times_ns','upstream_ccsdt_t3_to_t2',
        'upstream_ccsdt_map','upstream_multi_tsr_sym','upstream_fast_3mm',
        'upstream_fast_diagram','upstream_fast_sym_4d','upstream_fast_sym',
        'upstream_fast_as_as_sy_tensor_ctr','upstream_fast_sy_as_as_tensor_ctr',
        'upstream_fast_tensor_ctr','d4_memcontrol','d4_timer_util','d4_blas_flops',
        'dense_low_memory','d5_common','d5_value_interfaces','d5_algebra_interfaces',
        'upstream_fft_with_idx_partition','upstream_fft','upstream_dft_3d',
        'upstream_endomorphism','upstream_endomorphism_cust','upstream_endomorphism_cust_sp',
        'upstream_univar_function','upstream_bivar_function','upstream_bivar_transform',
        'upstream_test_suite_dense','upstream_matmul','upstream_recursive_matmul',
        'upstream_ccsd','upstream_ao_mo_transf','upstream_neural_network',
        'upstream_bitonic_sort','upstream_checkpoint','upstream_force_integration',
        'upstream_particle_interaction','upstream_qinformatics','upstream_mttkrp'
    )
    $denseArguments = @()
    foreach ($test in $denseTests) { $denseArguments += @('--test', $test) }
    foreach ($ranks in 1,2,4) {
        $env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = "mpiexec -n $ranks"
        cargo test @denseArguments
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Remove-Item Env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER
    cargo test --test scaling
    exit $LASTEXITCODE
}

$mpiTests = @(
    'foundation','dense_views','replicated_sum','tensor_sum','custom_reduce',
    'd4_memcontrol','d4_timer_util','d4_blas_flops',
    'd5_common','d5_value_interfaces','d5_algebra_interfaces',
    'upstream_fft_with_idx_partition','upstream_fft','upstream_dft_3d',
    'upstream_test_suite_dense','upstream_matmul','upstream_recursive_matmul',
    'upstream_ccsd','upstream_ao_mo_transf','upstream_neural_network',
    'upstream_bitonic_sort','upstream_checkpoint','upstream_force_integration',
    'upstream_particle_interaction','upstream_qinformatics','upstream_mttkrp',
    'algebra_sum','complex_scalar','sum_remap','replicated_contraction','ctr_2d',
    'tensor_gemm','algebra_contraction','tensor_contract','contract_remap',
    'dense_semantics','upstream_dense','subcomm_dense','plan_cache','model_training',
    'selector','selection_objective','tensor_blas_fold','upstream_gemm4d',
    'upstream_fast_3mm','upstream_fast_diagram','upstream_fast_sym_4d','upstream_fast_sym',
    'upstream_fast_as_as_sy_tensor_ctr','upstream_fast_sy_as_as_tensor_ctr','upstream_fast_tensor_ctr',
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
$arguments += @('--test', 'binary_io')
$arguments += @('--test', 'schedule')
$arguments += @('--test', 'cyclic_reshuffle')
$arguments += @('--test', 'bool_norm')
$arguments += @('--test', 'symmetric_reshuffle')
$arguments += @('--test', 'symmetric_subworld')
$arguments += @('--test', 'upstream_permute_multiworld')
$arguments += @('--test', 'indexed_write_order')
$arguments += @('--test', 'symmetric_permuted_io')
$arguments += @('--test', 'sparse_permuted_io')
$arguments += @('--test', 'upstream_scalar', '--test', 'upstream_speye')
$arguments += @('--test', 'upstream_ccsdt_map', '--test', 'upstream_ccsdt_t3_to_t2')
$arguments += @('--test', 'symmetric_random')
$arguments += @('--test', 'upstream_weigh4d', '--test', 'upstream_dft', '--test', 'distributed_symmetric_function')
$arguments += @('--test', 'integer_random')
$arguments += @('--test', 'distributed_sparse_dense_output')
$arguments += @('--test', 'distributed_sparse_storage_dispatch')
$arguments += @('--test', 'upstream_spmv')
$arguments += @('--test', 'distributed_sparse_reduce')
$arguments += @('--test', 'distributed_sparse_replicate')
$arguments += @('--test', 'upstream_scan')
$arguments += @('--test', 'distributed_sparse_2d', '--test', 'upstream_trace')
$arguments += @('--test', 'distributed_sparse_2d_dense')
$arguments += @('--test', 'distributed_sparse_2d_pairs')
$arguments += @('--test', 'distributed_coo_2d')
$arguments += @('--test', 'upstream_sssp')
$arguments += @('--test', 'distributed_mixed_coo')
$arguments += @('--test', 'upstream_strassen')
$arguments += @('--test', 'distributed_sparse_plan', '--test', 'upstream_spectral_element')
$arguments += @('--test', 'upstream_jacobi')
$arguments += @('--test', 'upstream_hosvd')
$arguments += @('--test', 'distributed_sparse_raw', '--test', 'distributed_sparse_search')
foreach ($test in @('upstream_subworld_gemm','upstream_readall','upstream_readwrite','upstream_sptensor_sum')) { $arguments += @('--test', $test) }
foreach ($test in @('typed_grid_blas','typed_randomized_svd','typed_distributed_eigh','typed_tensor_svd','typed_multilinear','tttp_memory','tensor_norms','storage_conversion','sparse_random_fill','subworld_transfer','sparse_text_io','symmetric_norms','symmetric_text_io','pair_read')) { $arguments += @('--test', $test) }
foreach ($ranks in 1,2,4) {
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = "mpiexec -n $ranks"
    cargo test @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Remove-Item Env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER
cargo test --lib tttp_blocking
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --lib sparse_text
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test narrow_algebra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test sparse_cost
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test sparse_mapped_cost
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test sparse_keys
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test sparse_coo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test mixed_kernel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test sparse_matricize
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test mixed_sparse_output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --test self_mapping
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = 'mpiexec -n 7'
cargo test --test upstream_strassen
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item Env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER
$localTests = @('scaling','random_generator','scalar_blas','node_peer_counts','folded_cost','partial_fold_kernel','fold_storage','partial_fold','fold_indices','fold_layout','fold_selection','mapped_cost','local_linalg','topology_candidates','node_aware','map_tensor',
    'sequential_sum','virtual_sum','sequential_contraction','folded_contraction',
    'sparse_formats','sparse_sequential','sparse_function','sparse_function_kernel','cost_models','plan_cost','grid_plan_cost','redist_cost','mapping_preflight','mapping_variants','topology_canonicalization','normal_mapping','symmetry_layout','sym_indices',
    'sym_triple','sym_operations','folding')
$arguments = @()
foreach ($test in $localTests) { $arguments += @('--test', $test) }
cargo test @arguments -- --nocapture
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --lib schedule
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --lib cyclic_reshuffle
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --lib symmetric_reshuffle
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --lib sparse_virtual
exit $LASTEXITCODE
