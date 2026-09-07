# Tensor-pair / COO matricization

`sparse_matricize` ports sparse_formats/coo.cxx set_data/get_data rather than
converting through dense storage. Forward mapping divides global coordinates
by their phases, inverts the source reverse-ordering permutation, and groups
the reordered dimensions into matrix rows and columns. One-based coordinates,
input pair order and values are retained.

When folded dimensions are fewer than tensor axes, forward conversion uses
the source rising-binomial rank within non-NS groups. The source uses the
same formula for SY, AS and SH; this is not replaced with a different strict
antisymmetric packed rank. Callers supply the source folded lengths and
padded local dimensions.

Reverse conversion is the defined unfolded get_data algorithm: restore
phase residues, encode full tensor keys, then sort by key. Unlike depin it
does not filter phase padding. The source marks folded symmetric reverse
conversion FIXME, so no invented folded reverse API is exposed.

These are local layout algorithms for subsequent sparse plan assembly;
automatic selection and wiring of every tensor-folding route remain pending.
