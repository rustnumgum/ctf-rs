# Sparse CPU cost estimates

`sparse_cost` ports the fixed version's explicit 2D, replicated and virtual
layer estimates; `sparse_cost::local` ports CPU sequential kernel IDs 0–5.
They use the existing named `cost::Models` bank. Model evaluation does not
communicate, allocate tensor storage, time a kernel or modify coefficients.

`KeyPinning` also retains source pin_keys_mdl timing (two passes for C) and
the literal memory switch fallthrough described in sparse-keys.md.

## Local kernels

The general leaf multiplies the distinct union-label extents; folded leaves
use m*n*k. Each sparse input multiplies the source flop estimate by its
nonzero fraction times three. Sparse output does not change the flop count.
The source traffic heuristic scales sparse A/B bytes by fraction*10 and
sparse C by fraction*30, truncating each operand before summation. Folded
packed-element counts exclude the folded axes, which the source constructor
replaces by unit extents, before multiplying by m*k, n*k and m*n.

All twelve CPU ordinary/custom named models are selected explicitly. CUDA
offload models are excluded, as required. A custom model's existence does not
imply that every storage/function combination is a supported execution leaf.

## Communication and memory

Sparse 2D panel sizes count virtual blocks and include dense virtual-block
size; dense sizes count elements. These are source model units, not uniformly
the matrix-block units of the Rust executor's Panel. Sparse reductions use
csrred_mdl even for custom addition, retaining the pinned source selection.

The 2D child time is estimated at one layer before multiplying by
edge/min(layers,edge). Replication retains the parent's layer count; virtual
execution multiplies child time by the product of virtual dimensions.
The standalone layer APIs accept child estimates explicitly. The raw-mapping
builder below now assembles these source layers without a backend framework.

Resident and temporary costs retain the source's distinct conditions:
sparse input copies for moving/strided 2D panels, three sparse output buffers,
replicated sparse inputs only with more than one input communicator, and
replicated dense output temporary storage only when it is reduced.
The recursive memory rule is footprint + max(temporary, child memory), except
the virtual layer, which simply adds its source integer-index bookkeeping.

These estimates are source heuristics, not measured Rust peak memory or
guaranteed runtime. Automatic sparse candidate-plan cost attachment remains
unfinished; no performance ratio is claimed from a prediction.

## Original-layout density and redistribution

`Fractions::from_layouts` ports contraction.cxx:211-242,2620-2629. Sparse
counts are divided by padded local size and the product of physical mapping
phases, not by all context ranks (which can include replicas). Counts include
stored zeros. The caller provides globally counted canonical entries explicitly.
For sparse C the source estimate is the larger of existing density and
A-density * B-density * contracted-label volume, capped at one. A user output
fraction overrides that estimate before the final cap. Dense storage defaults
to density one. This does not evaluate tensor values or communicate implicitly.

`redist_cost::sparse` ports untyped_tensor.cxx:25-28,3196-3245, with the same
outer unchanged-mapping preflight as the existing dense API. Sparse storage
never uses block reshuffling, even at equal phases. Model traffic truncates
element_size * max(old_size,new_size) * fraction before multiplying by log2(np);
temporary memory truncates pair_size * max_size * fraction * 2 afterward.
Neither formula is a measured allocation count.

## Unfolded raw-mapping assembly

`sparse_mapped_cost::build_unfolded` constructs the source sequence from actual
distributions: key pinning, missing-axis replication, shared-index 2D panels,
residual virtual blocks, and the general k0 leaf. Sparse mutable panel state
counts virtual blocks; dense state counts elements. The builder retains physical
head phases, source operand orientation, outer/inner strides, and LCM panel
edges. It accepts preflight-valid mismatched shared mappings, not only aligned
GridPlan layouts. Its scope is NS unique-label sparse A with dense B/C; A-only
labels are rejected as in the executable general recursion.

This is `construct_sparse_ctr(is_inner=false)` (contraction.cxx:3951-4354),
not folded k1-k5 selection. Folded kernels require their own fold metadata.
The source comment calling k0 obsolete is not an assertion removing that branch.
`Plan::estimate_with_redistribution` adds the source changed-input residency,
sum of redistribution temporaries, output round-trip time, and recursive
working-memory maximum. It is cost metadata, not another cached execution plan.

Three exact local checks cover aligned replication/virtualization, a normal
GEMM mapping with mismatched shared axes and sparse panel strides, and the
supported input boundary. The unfolded sparse-A estimates now feed
normal/exhaustive selection and raw panel execution (see sparse-search.md).
Folded and sparse-output candidate integration remains unfinished; metadata
tests alone do not prove execution.
