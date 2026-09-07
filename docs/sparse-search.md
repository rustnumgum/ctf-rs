# Unfolded sparse search and execution

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

This connects selection to execution for the unfolded unique-label sparse-A
path, not all sparse planning. Folded k1-k5 candidate assembly/selection, sparse
B/C output combinations, compressed-symmetry automatic plans and search-cache
integration remain pending. Existing aligned GridPlan caching remains available;
it is not mislabeled as a cache for this arbitrary raw search.
