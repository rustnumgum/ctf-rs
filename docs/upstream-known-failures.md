# Reproduced pinned-source failures

## Bounded S1c sparse checkpoint precision (dense-twin finding)

The source writes f64 sparse coordinates with six decimal places, but the
bounded n=3 checkpoint criterion is `norm2(v-u) < 1e-7*n*n*.1*n`, or 2.7e-7.
One prescribed WSL run each gave 3.9769591334653147e-7 at one rank and
3.4024306287098844e-7 at two ranks. Four ranks passed the world assertion
(its numerical value was not printed) and failed the parity assertion with
3.4024306287098844e-7. No value, fixture, threshold or text precision changed.

One permitted dense-twin diagnostic at one rank generated the same sparse
fixture, converted it to dense storage, and ran the same text write/read and
subtraction through dense APIs. It gave the identical 3.9769591334653147e-7
failure. This separates sparse selected-sum/key storage from the source-format
loss in this bounded expectation; it is not a new C++ runtime reproduction.
The driver remains unaccepted. Diagnostic budget: 1/3, no further study.
Evidence: `D:/projects/runs/ctf-rs-s1/S1c/upstream_checkpoint_sparse-<ranks>.log`
and `checkpoint-dense-twin.{rs,sh,log}`. The final native results are recorded
in `validation.md`; the source file format and acceptance rule remain fixed.

## S1a sparse Python ABC expressions (static source restriction)

Pinned `test_sparse.py` expects sparse `ijk,jkl->ijkl` and `ijl,kjl->ijk`
expressions to execute. Their shared ABC labels violate sparse `can_fold`
(`contraction.cxx:537-584`); non-inner construction asserts dense B and C
at `contraction.cxx:4313-4335`. The Rust raw search preserves these rules.
S1a WSL 1/2/4 once each returned no eligible plan and stopped before Q could
be computed in both semantic drivers. This is a static source contradiction,
not a reproduced C++ runtime failure or numerical PASS. The strict source
sum(abs(diff)) < 1e-14 rule is unchanged; neither test is accepted.
Logs: `D:/projects/runs/ctf-rs-s1/S1a/sparse_{einsum_hadamard,scaled_expression}-<ranks>.log`.
Later GEMM-shaped S1b/S1c work does not require these source-forbidden labels.

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
