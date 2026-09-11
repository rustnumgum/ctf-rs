# Fixed-source automatic planning integration

## Node-count configuration boundary

Pinned CPU topology construction takes an explicit processes-per-node value:
default ppn=1, with CTF_PPN as a source override. It requires ppn>=1 and exact
world-size divisibility; world.h documents consecutive same-node ranks.
get_phys_topo/get_generic_topovec factor process counts rather than discovering
hardware, and node-aware grid selection receives an explicit node count.
Rust's explicit ranks_per_node selection input preserves this boundary.
Context::split_shared is available for callers that want MPI shared-memory
groups, but automatic hardware discovery is not a missing pinned CPU routine
to invent as part of this port. Sparse node-aware integration remains pending.

Reference: cc4s CTF f69cbb46e23bc2f39cda5722ce096f56301dab4f.

## Collective raw dense-unfolded search

dense_search::search_dense connects source normal and exhaustive
candidate generation, raw mapping costs, rank-local selection, global winner
selection and selected-ID reconstruction. It does not constrain candidates to
GridPlan's aligned maps. Old topologies and catalog topologies have explicitly
supplied node-count facts; equal topologies must have equal facts.

The normal search first minimizes time. A non-negligible memory weight triggers
the source second normal pass using the first winner as baseline. Optional
exhaustive refinement starts only at normal time >= 0.01, uses that normal winner
as baseline, and is accepted only when its time is strictly lower, even for a
weighted search. Strict mapped-residency/detailed-memory bounds and INT_MAX local
element limits precede ranking. Equal scores retain source local order and then
rank order. Only local winner costs and the winning ID are communicated, not all
candidate maps or tensor values.

SearchCache retains these selected mappings using the structural contraction
signature and old node facts. Context, catalog, model coefficients and search
options are fixed for the cache lifetime; tensor values and alpha/beta are absent
from the key. Misses search collectively, hits and clear are local, so ranks must
use the same call sequence. No collective occurs in destruction.

Tensor<A>::contract_from_mapped (A: Semiring + Clone, Element: Wire) executes the selected raw layout:
redistribution, outer replication, nested source 2D panel steps, virtual traversal,
the unfolded sequential kernel, output reduction and original-layout restoration.
Callers can feed SearchCache::prepare's selected distributions directly to it.
It does not convert 2D choices to aligned maps or globally gather tensor values.
The shared panel executor uses ordered MPI user reductions for arbitrary algebra
elements, including noncommuting matrix products. Set custom_reduce=true when
modeling this reduction implementation; native predefined-MPI-op specialization
is not yet selected automatically. Folded BLAS selection/execution, sparse
estimates/execution, selection-scan diagnostics and custom bivariate functions
remain unfinished here.

## Local dense fold permutation selection

folding::Plan::select now implements all six source dense fold permutations and
their GEMM transpose/output-swap flags. It minimizes modeled transpose time with
per-operand virtual multiplicities, and source ties choose the last candidate.
The noncommutative restriction considers only the first three permutations.
Plan::new still explicitly represents permutation zero. Both execute through the
compile-time LocalKernels boundary (native BLAS initially).

This is the fully-foldable local NS stage, not yet map_fold's full distributed
partial/symmetry folding or its integration into automatic total-cost search.

fold_indices::select now supplies the source get_fold_indices/can_fold stage for
NS/SY/AS/SH links, including partial groups, repeated-label/dense-custom early
rejection and sparse restrictions. fold_layout::FoldLayout supplies tensor::fold
metadata from local virtual-block lengths: packed symmetry-group lengths, the
selected rec_tsr shape and index IDs, and selected-first inner ordering. Its
permutation changes only the selected prefix. These metadata stages do not yet
transpose distributed packed storage or execute partially folded contractions.

FoldLayout::transpose now converts owned local storage forward/backward for each
virtual block, treating compressed symmetry groups as packed dimensions rather
than expanding them. partial_fold::select composes eligibility, packed group
layouts and all six source permutations using full local group lengths, including
the residual/unfolded dimensions in transpose costs. Its descriptor exposes the
packed lnmk parameters, transpose flags and selected storage ordering. The source
partially folded local kernel and distributed folded execution are still pending.

partial_fold_kernel::execute<K> now performs one local f64 partial-fold block:
owned storage transpose, selected-dimension collapse, source residual-index
traversal and packed offsets, beta-once scaling, repeated batched BLAS calls and
inverse C transpose. This is the canonical inner kernel; symmetry multiplicity,
diagonal prescaling and distributed folded orchestration are not supplied by it.
AS/SH fold groups use source sy_packed_size capacity, retaining diagonal holes,
not the smaller logical packed_size domain.

folded_cost::estimate_dense_folded now supplies the raw dense NS folded estimate:
the existing replication/panel/virtual communication tree surrounds a source
folded BLAS cost leaf, with partial-fold transpose time and all mapped tensor fold
buffers included. Memory is redistributed-input residency plus the maximum of
redistribution temporary storage and fold buffers plus inner workspace. Ineligible
folds return None, not a substituted estimate. Automatic search still explicitly
uses its unfolded estimator; selecting/executing folded distributed candidates is
not yet connected by this cost API alone.

The integrated search now accepts explicit Options.enable_folding. When enabled,
ordinary eligible candidates use the folded estimator in every normal, weighted
and exhaustive pass. Source-ineligible folds and dense custom functions use the
source unfolded branch; estimation failures are not converted into another plan.
Selected.fold carries the winning descriptor, rebuilt after selected-ID mapping
reconstruction without repeating its full cost estimate. SearchCache retains it
under the same immutable configuration. Callers must distinguish this folded
selection from the existing raw unfolded executor; distributed folded execution
is still pending. The old search_dense_unfolded name has been removed, not wrapped.

## Executable inner cost tree now connected

GridPlan::cost_tree derives the aligned unfolded dense execution tree from the
actual mapped layouts: optional replication, optional virtualization, local
contraction. It no longer requires callers to fabricate a matching Tree.

- Local operand bytes: product(block_shape)*element_bytes; source ctr_tsr.cxx
  407-418 and untyped_tensor.cxx 640-669.
- Local FLOPs: twice the product of every unique local label extent; ctr_tsr.cxx
  421-441. Standard unfolded local model, not the folded BLAS/custom-function model.
- Virtual calls and bookkeeping: ctr_tsr.cxx 82-105, using one phase per label.
- Missing-input fibers broadcast; missing-output fibers reduce. Messages use
  full mapped local storage bytes. Wholly unused axes are skipped; ctr_comm.cxx
  43-115 and 173-249. Node counts are explicitly supplied, not guessed.

This estimate excludes input/output redistribution, folding, tensor residency
and Rust allocator overhead. Its working_bytes is the source recursive workspace
model, not peak RSS. Do not feed this inner estimate into Selector as total plan
cost or call it completed automatic selection.

## Unfolded dense redistribution now connected

GridPlan::estimate_unfolded adds the pinned dense redistribution models to the
inner tree. Equal per-axis total phases select blres_mdl; otherwise dgtog_res_mdl
uses process log and maximum old/new local bytes. Unchanged maps cost zero.
Temporary memory is max local bytes for block reshuffle and floor(1.5*bytes) for
general dense reshuffle. See untyped_tensor.cxx 3196-3244.

detail_estimate_mem_and_time's unfolded branch counts changed A/B mapped residency,
sums A/B/C temporary buffers, doubles C redistribution time, and uses
resident_AB+max(redist_temporary,inner_workspace). These values are exposed in a
detailed estimate rather than conflated with measured RSS. Source topology pointer
identity is represented by Rust structural topology equality.

This covers ordinary dense redistribution without offset/permutation arguments.
It models pinned source algorithms, not the performance of the current key-routed
Rust I/O implementation. Folded/sparse/panel alternatives and automatic candidate
enumeration remain incomplete.

## Remaining source construction and selection steps

The current GridPlan creates one globally aligned greedy map. Upstream normal
search instead evaluates six operand orders, old-layout subsets and each topology
(contraction.cxx 2834-2913). Category-specific assignment is at 2392-2553 and
check_mapping preflight at 1339-1572. Those must precede candidate evaluation.

Exhaustive-map combinatorics are at 2008-2041, construction at 2143-2388 and
distributed enumeration at 3031-3105. Exhaustive search is a refinement, not a
fallback after failure: it is skipped below the source 0.01-second threshold and
wins only when strictly faster (3311-3338).

VariantSpace now implements the NS unique-label count and raw decode stages,
including all three 2D orientation values and shared-label LCM virtualization.
Candidates retain differing physical maps across operands. The separate
mapping_preflight::check ports source phase and physical-mismatch validation for
unique-label NS distributions, including physical-chain self checks. Neither
component claims folded/symmetry self-mapping validation or executes a candidate.

Pinned quirks are intentional: hollow get_choice index zero yields repeated zero
coordinates, and AC orientation case 1 has no assignment because its source break
precedes the assignment. These are not replaced with textbook combinations/SUMMA.
Variant::canonicalize now applies source folded-pair topology permutation and
first matching catalog lookup. visit_local_exhaustive streams decode -> canonicalize
-> preflight per rank. Raw global IDs include skipped candidates and partition by
ID modulo communicator size; no survivor renumbering occurs. Remaining integration
includes size/memory filters and costed execution.

normal_mapping::Problem now ports the NS unique-label category helpers and both
mapping passes in map_to_topology. The source six operand permutations retain
different physical maps for paired contraction dimensions. Initial mappings
determine padding; physically mapped common/singleton labels are rejected as in
the source. Returned distributions are raw candidates and require preflight
before tensor use, since intermediate copying can duplicate physical axes.
normal_search::visit_local implements the seven old-layout subsets followed by
catalog topologies, partitioned by (template-1) modulo communicator size. It
preserves source ID 6*template+permutation and rejects failed mappings/preflight
without renumbering. This enumeration does not yet perform cost/memory selection
or replace GridPlan's aligned-only preparation.

Selected-ID reconstruction is available for both searches. Normal IDs decode
template/permutation and remap the chosen retained subset or catalog topology.
Exhaustive IDs locate the raw cumulative-count interval, decode just that variant
and apply canonical topology permutation. Neither path requires enumerating other
maps or broadcasting tensor contents. Final preflight is mandatory, matching the
source assertion for a previously accepted selected ID.

mapped_cost::dense_unfolded now builds raw NS mapping cost trees, including
label-ordered 2D panels with input broadcasts or output reductions, source
per-operand block-state updates, residual virtualization and outer replication.
estimate_dense_unfolded adds dense redistribution and temporary/resident memory
terms. The stationary panel operand has zero panel-buffer bytes; it is not counted
as a copied full tensor. Communicator node counts are explicit inputs.

Total candidate evaluation must additionally port folding and sparse redistribution
costs (2632-2810), then connect existing collective Selector and context cache. Sparse,
symmetry, node-aware, low-memory and 2D panel alternatives remain in the overall
scope; cost-tree availability is not proof of integrated candidate execution.

## Selected distributed folded execution

`Tensor<Arithmetic<f64>>::contract_folded_from_mapped<K: LocalKernels>` now
executes the raw distributions and fold descriptor returned by `SearchCache`.
Each original local virtual block is transposed once before replication and
nested 2D panels. Virtual leaves operate directly on packed blocks; output is
inverse-transposed after reduction and restored to its original distribution.
This supports dense NS partial folds and batch labels without forcing aligned
maps or globally gathering tensors. Sparse/compressed automatic execution
remains unfinished. `LocalKernels` retains
the compile-time replacement boundary for future faer kernels.

## Node-aware dense folded execution

`node_reordering::select_dense` evaluates the source inter-node grid candidates,
retains the first strict volume minimum and enables reordering only on strict
improvement. Its chosen intra-node lengths can now be supplied to
`contract_folded_from_mapped`. Execution creates the rank-reordered context,
uses MPI_Sendrecv_replace on local packed blocks, runs the existing raw panel
tree in that context and backmaps C before restoring its distribution. Private
A/B copies are discarded rather than unnecessarily exchanged back.

Node facts are explicit f64 average peer counts throughout raw/aligned cost
trees, folded costs, search and node selection. `original_peer_counts` ports
the source node-boundary average without rounding; cache keys preserve the
supplied facts' bit patterns. Automatic hardware-node discovery is not yet
integrated. Direct sparse node-aware execution is excluded by the pinned
`ppn != 1 && !is_sparse()` branch, not an unfinished working CPU capability
(see source-node-aware-boundary.md). Logical node partitions in
single-machine tests do not establish multi-node performance.

## Low-memory folded execution

`contract_folded_low_memory` takes distinct mutable A/B/C tensors rather than
retaining cloned home data. Its shared folded executor redistributes and packs
the live buffers, then restores C, A and B to their original distributions.
Node-aware execution also backmaps A/B because the originals must survive.
This is the ownership counterpart of lowmem_contract, not an alternate
contraction algorithm. Sparse tensors/custom functions are not accepted by
this f64 dense API; global zero-length dimensions retain the source rejection.
Weighted search remains explicit through Options.weight, so callers may reuse
the selected plan without embedding collective planning in destruction.

The immutable-input folded API retains its private working copies. Low-memory
execution removes those retained home copies but still allocates redistribution
and transpose workspace; it is not a zero-allocation or hard-RSS-bound promise.
The representative mpi_low_memory_bench example measures one contraction with
search excluded and both redistribution/restoration included.

`contract_low_memory_from_mapped` supplies the same live-buffer ownership policy
to the raw, unfolded generic semiring executor. It retains ordered MPI user
reductions, right alpha scaling, nonscalar left beta and the scalar/no-replication
right-beta special case. Mutable operands restore their original distributions
in C/A/B order. This adds integer, f32, complex and custom-algebra low-memory
execution without routing them through f64 BLAS or introducing a backend layer.
Both generic raw entry points also accept optional intra-node lengths. The
node-aware context permutation and forward/back exchanges use the same source
rank rules as the folded path. Generic values are encoded into one Wire byte
buffer for MPI_Sendrecv_replace and decoded afterward; Rust object layout is
never treated as an MPI datatype. This communication workspace is distinct
from the retained home copies eliminated by low-memory execution.

## Explicit sparse planning configuration (S1)

`sparse_search::SearchCache` selects raw k0-k5 contraction mappings with
per-operand `StorageSize`; selected ordinary and heterogeneous custom methods
execute those distributions directly. `sparse_symmetric_search` selects
compressed sparse custom accumulation layouts, consumed without a dense force
matrix. `sparse_sum_search::SearchCache` selects source two-pass summation
mappings with pin/permutation/replication/virtual execution metadata.
Drivers explicitly own catalogs, Models and memory options; old direct-key
sum APIs do not silently construct defaults. Cache signatures omit values and
nonzero counts; a miss is collective and a structural hit reuses the plan.
Sparse folding and sparse-output replication rejection remain source rules.
Acceptance and the source-restricted HANDOFF cases are in `validation.md`.

## Typed native BLAS boundary

`Gemm<T>` and `GemmKernel<T>` carry the scalar type through the local matrix
kernel, with native SGEMM/DGEMM/CGEMM/ZGEMM implementations. `LocalKernels`
requires `GemmKernel<f64>` and continues to describe existing f64 LAPACK
operations; no unimplemented f32/complex decompositions are advertised.
Local contiguous-batch and partial folded residual kernels now accept typed
arithmetic values directly. Distributed folded orchestration now carries the
same scalar type through virtual blocks, raw MPI panels, node rank reordering
and low-memory restoration. Public folded entry points retain a single kernel
type parameter, with the scalar inferred from Tensor<Arithmetic<T>>. Planning
uses the actual element width rather than assuming eight-byte values.
Output reductions use the existing ordered generic MPI operation, so callers
model this executor with custom_reduce=true; no scalar promotion is performed.
The compile-time traits remain the replacement boundary for future faer kernels.
