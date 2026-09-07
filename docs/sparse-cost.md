# Sparse CPU cost estimates

`sparse_cost` ports the fixed version's explicit 2D, replicated and virtual
layer estimates; `sparse_cost::local` ports CPU sequential kernel IDs 0–5.
They use the existing named `cost::Models` bank. Model evaluation does not
communicate, allocate tensor storage, time a kernel or modify coefficients.

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
Callers supply these child estimates explicitly rather than using another
backend or plan-tree framework.

Resident and temporary costs retain the source's distinct conditions:
sparse input copies for moving/strided 2D panels, three sparse output buffers,
replicated sparse inputs only with more than one input communicator, and
replicated dense output temporary storage only when it is reduced.
The recursive memory rule is footprint + max(temporary, child memory), except
the virtual layer, which simply adds its source integer-index bookkeeping.

These estimates are source heuristics, not measured Rust peak memory or
guaranteed runtime. Automatic sparse candidate-plan cost attachment remains
unfinished; no performance ratio is claimed from a prediction.
