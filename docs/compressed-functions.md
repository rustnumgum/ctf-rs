# Compressed custom-function contraction

Pinned source: `f69cbb46e23bc2f39cda5722ce096f56301dab4f`.
`SymmetricTensor::contract_function_from_on` now threads a homogeneous local
binary function through the packed CPU/MPI path. It takes the ordinary explicit
mapping/alpha/beta/MPI-commutativity arguments followed by the function.
Ordinary multiplication shares the same generic implementation; no dense
unpack fallback or additional backend is introduced.

## Implemented literal port

A local function reference passes through existing orchestration:

1. `symmetric_contract.rs`: repeated-index extraction/reinsertion,
   `contract_sy_recursive`, `contract_raw_on`, and prescaled operand branches.
2. `symmetric_contract_tensor.rs`: canonical mapping, packed redistribution and
   restoration, retaining the existing communication contexts.
3. `symmetric_contraction_comm.rs`: replicated and virtualized execution.
4. `symmetric_contraction.rs`: packed sequential function evaluation at exactly
   the current multiplication offsets.

The required bounds remain `Group + Semiring + Clone + CastFromF64` and `Wire`
elements. The function itself executes locally and needs no serialization.
MPI reduction commutativity is not commutativity of the binary function.

`ctr_tsr.cxx:534-557` dispatches to `sym_seq_ctr_cust`;
`sym_seq_ctr.cxx:241-264` computes f(A,B), multiplies alpha on the right, then
adds contribution before old C. Keep whole packed-C beta scaling and scalar
special cases. `contraction.cxx:5025-5049` gates input-only self-reductions on
function distributivity. Rust currently has no such optimization, so retaining
the joint traversal requires no invented distributivity assumption.

## Source quirks are not stronger mathematical guarantees

Preserve AS alignment/permutation signs and symmetry overcount factors in alpha
(`contraction.cxx:5115-5157,5219-5233`). Preserve coincidence-mask diagonal
prescaling of operands before calling the function (`:5800-5992`). Nonlinear
functions consequently see prescaled operands; arbitrary mathematical
equivariance under these transformations is not guaranteed by the source.
Do not silently move factors inside f, change signs, reject working source
cases, or claim a corrected algorithm as a literal port.

Custom low-memory contraction is explicitly rejected by the source
(`contraction.cxx:5685ff`); custom folded/inner paths require additional kernel
capabilities. Neither is implied by implementing the packed scalar CPU path.
