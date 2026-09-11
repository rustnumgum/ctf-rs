# Reproduced pinned-source failures

## Related sparse_permuted_slice expectation (static finding only)

The callback-copy expectation in `examples/sparse_permuted_slice.cxx` has the
same full-orbit/canonical-write multiplicity conflict. Each off-diagonal child
entry is requested in both orientations. SY/SH accumulate x+x; AS applies the
parent-read sign and destination-write sign, also accumulating x+x. Thus a
nonzero compressed callback becomes 2x rather than x. This is a static inference
from the source-compatible transfer steps, **not an executed result for this
example**. Its draft Rust driver was not registered or run; no normalization,
tolerance relaxation, or branch removal was used to claim a pass. The example
remains unaccepted. The source order-three singleton callback's `ij` notation
would use explicit `ijk` in Rust without preserving C++ string ABI behavior.

These are failures of the unmodified C++ reference, not Rust acceptance passes.
Do not relax tolerances or change the Rust semantics merely to hide them.

## permute_multiworld: compressed expected-copy assertion

On 2026-09-07, `/home/xylxp/ctf-rs-reference` reported HEAD
`f69cbb46e23bc2f39cda5722ce096f56301dab4f` and an empty `git status --short`.
The static library was built out-of-tree with GNU C++ 15.2, MPI/OpenMP, shared
libraries/tests/ScaLAPACK disabled. The unchanged `test/permute_multiworld.cxx`
was linked with that library and OpenBLAS. All C++ build artifacts reside only
under `/home/xylxp/.cache/ctf-reference-build`, outside the Rust build.

The single diagnostic run was:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun --oversubscribe -n 1 \
  /home/xylxp/.cache/ctf-reference-build/permute_multiworld -n 3
```

Observed output:

```
Testing nonsymmetric multiworld permutation with n=3
{ permuted read and write among multiple worlds } passed
Testing symmetric multiworld permutation with n=3
{ permuted-read among multiple worlds } failed
Assertion `pass' failed. (test/permute_multiworld.cxx:194)
```

The executable aborted in the SY branch. AS/SH branches were not reached, and
no further rank or precision runs were performed. This is sufficient to reject
the hypothesis that only the Rust rewrite introduced this driver discrepancy.
It does not assert that unexecuted branches have been verified.

Source tracing separately explains the incompatible contracts: the driver
expects an unchanged blocked copy, while `tensor::permute` calls full-orbit
`read_local(..., true)` for a compressed destination and subsequently writes
those entries through signed canonical duplicate accumulation. The Rust
implementation preserves that source path and tests it explicitly. A normalized
copy would be a deliberate behavior change, not a faithful fix to the port.

Stop That Digit: one discriminating baseline computation; known upstream FAIL.
This does not reopen passing Rust numerical checks or add C++ runtime/build
dependencies to the delivered crate.
