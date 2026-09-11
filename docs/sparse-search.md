# Raw sparse search and execution

S1a adds `SearchCache` with fixed model/storage configuration and structural
cache keys, global canonical nonzero counts on misses, and local hits.
`Selected` retains the actual distributions, sparse fold descriptor and COO
capability. Folded k1-k5 candidates use sparse density/transpose, twelve CPU
leaf models and A/B/C pin layers. Selected execution performs redistribution,
folding, replication, nested panels, virtual traversal and output depinning;
it does not substitute a GridPlan. APIs cover SDD, SSD, SSS and SDS storage.
Sparse ABC weigh labels remain source-ineligible. Compressed-symmetry and
custom mixed-type selected execution remain S1b; summation remains S1c.

`sparse_search::search_unfolded` selects raw mappings for NS sparse A / dense
B / dense C. It reuses the existing source normal and exhaustive enumerators,
weighted objective and collective winner protocol. Nonzero counts refer to
canonical stored entries in the original layout, including stored zeros but
excluding replicas. Sparse time/workspace come from the sparse tree and sparse
redistribution formulas, not from the dense cost model.

The source preliminary element-byte check, strict memory bound, dense-only
INT_MAX local-count restriction, weighted second pass, .01-second exhaustive
refinement threshold, stable strict-minimum ties, and final time-based refinement
decision are retained. The result uses the existing selected-distributions
record with no folded descriptor. Selection is an explicit collective; this
API adds no separate cache, backend registry or implicit destructor operation.

`Tensor::contract_sparse_from_mapped` executes those raw distributions directly.
The same mapping traversal constructs cost and execution metadata, including
shared-index physical-axis mismatches, source panel strides, replication axes,
and residual virtual dimensions. Dense element strides are converted to whole
local virtual-block units, not arrays of single-element vectors. Per-level fiber
communicators are created once before recursive execution and closed explicitly
afterward. Sparse keys are pinned, broadcasts and nested `execute_pairs_dense`
panels feed the k0 leaf, and C is reduced and restored to its original layout.
Neither input nor output is globally gathered.

The execution entry point consumes structural descriptors only: density fields
do not influence that assembly, and no cost estimate is evaluated during it.
Search separately computes actual density-based estimates.

The older `search_unfolded` entry remains explicitly restricted to k0 SDD.
The S1a selected APIs described above extend it without changing that contract.
Existing aligned GridPlan caching is distinct from the raw SearchCache.
