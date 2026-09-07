# Fixed-source automatic planning integration

Reference: cc4s CTF f69cbb46e23bc2f39cda5722ce096f56301dab4f.

## Collective raw dense-unfolded search

dense_search::search_dense_unfolded connects source normal and exhaustive
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
