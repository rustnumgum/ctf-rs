# Sparse grid-plan reuse

`Tensor::contract_sparse_with_plan` executes sparse A / dense B / dense C
using the existing `planning::GridPlan` and context-borrowing `PlanCache`.
It consumes the plan's mapped distributions directly. Ordinary and custom
explicit-grid calls share the same mapped execution worker, including key
pinning, virtual blocks, input broadcasts, output reduction, and restoration
of the caller's distribution. Created fiber communicators close explicitly.

The reference is cc4s CTF f69cbb46e23bc2f39cda5722ce096f56301dab4f,
`contraction_signature.{h,cxx}` and the cache lookup in `contraction.cxx`.
Its signature records lengths, symmetry, indices and mappings, not sparse
flags, nonzero fractions, coefficients, algebra or custom-function identity.
On a hit the source reuses topology/exhaustive-selection metadata and builds
the computation from current operands. Rust likewise caches no tensor values
or leaf state, but uses structural equality rather than hash-only equality.

This API is limited to the existing NS tensor types: Rust's current signature
does not encode compressed symmetry. Unique labels per operand and no A-only
label are required by this raw sparse path. Physically mapped labels must
occur in at least two operands. Unsupported plans are rejected, not silently
remapped. Explicit nontrivial virtual factors remain available through `_on`;
they are not represented by `PlanCache::prepare` and are not claimed cached.
Automatic sparse/compressed candidate construction remains unfinished.

`distributed_sparse_plan` checks exact integer results while reusing a plan
after changing stored values/nonzero count to an empty sparse input, switching
requested topologies, and restoring the original output distribution. It also
checks cache hit/miss counts in world and parity subcommunicators.
