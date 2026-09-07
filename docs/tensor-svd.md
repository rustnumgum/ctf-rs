# Indexed tensor SVD and sequential HOSVD

`Tensor::tensor_svd` retains the dense truncated/randomized interface.
`SparseTensor::tensor_svd_truncated` provides the fixed source's native
truncated path for f32, f64, Complex<f32>, and Complex<f64>. Both validate the
input/output label partition, matricize in left-then-right non-auxiliary index
order, and restore the requested factor-axis ordering. Sparse input permutation
and reshape operate on stored keys; dense storage is materialized only for the
distributed native matrix decomposition. There is no full-tensor gather and no
C++ dependency. Inputs remain unchanged, and factors borrow the same context.

Sparse randomized SVD is not implemented by this new API. It must retain the
source sparse products rather than silently call the dense randomized path.

`tests/upstream_hosvd.rs` ports the working `examples/hosvd.cxx` algorithm,
not the unfinished C++ HoSVD class: four successive indexed SVDs retain ranks
R+3, R+2, R+1, R; the current core is multiplied by its singular values after
each step. The core and four right factors are contracted back into the input
shape. No eigenvector or singular-vector phase is compared.

The bounded fixture uses n=2, R=1 and R=2, all four scalar types, dense storage
at fraction 1 and sparse storage at fraction .8. It uses the existing explicit
rank-local random generator instead of a process-global RNG lifetime. Complex
bounds are real-only as in the source. The original acceptance quantity is
the reconstruction Frobenius norm, bounded by
`input_norm * (1 - (R/n)^4) + 1e-4`, with finite results required.
