# Node-aware source boundary

The pinned contraction.cxx:4696 explicitly guards node-aware candidate
selection with `ppn != 1 && !is_sparse()`. is_sparse() at lines 401–403 means
any sparse operand, not just sparse C. Pre/post rank exchange is likewise
dense-only (4733–4739 and 4847–4855). Sparse communication classes inherit
the zero volume estimate from ctr_comm.h:204–205 and do not override it.

Consequently direct sparse node-aware execution is **not a working CPU
capability of this revision**. Adding it would require new sparse payload,
metadata and ownership algorithms, contrary to the direct-port boundary.
Earlier inventory wording “sparse node-aware integration pending” incorrectly
made that unsupported enhancement appear to be a required source port.

Dense node-aware execution remains in scope and implemented separately.
When the source converts sparse C to dense storage before contracting, that
is a dense operation and may use the existing dense node-aware path. Automatic
sparse plan assembly is still genuinely unfinished; this boundary does not
resolve or narrow that requirement.
