# Fixed-source automatic planning integration

Reference: cc4s CTF f69cbb46e23bc2f39cda5722ce096f56301dab4f.

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

## Remaining source construction and selection steps

The current GridPlan creates one globally aligned greedy map. Upstream normal
search instead evaluates six operand orders, old-layout subsets and each topology
(contraction.cxx 2834-2913). Category-specific assignment is at 2392-2553 and
check_mapping preflight at 1339-1572. Those must precede candidate evaluation.

Exhaustive-map combinatorics are at 2008-2041, construction at 2143-2388 and
distributed enumeration at 3031-3105. Exhaustive search is a refinement, not a
fallback after failure: it is skipped below the source 0.01-second threshold and
wins only when strictly faster (3311-3338).

Total candidate evaluation must additionally port redistribution/folding costs
(2632-2810), then connect existing collective Selector and context cache. Sparse,
symmetry, node-aware, low-memory and 2D panel alternatives remain in the overall
scope; the aligned-only tree does not substitute for those branches.
