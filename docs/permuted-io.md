# Coordinate permutation between worlds

Dense tensors provide `gather_permuted_into` and `scatter_permuted_from`. The
parent tensor is `self`; every rank of its context calls the operation. Each rank
may supply an optional child tensor. Unlike add_to/from_subworld, different child
contexts may participate in the same call, as in upstream `permute_multiworld`.
All members of a child must participate consistently. No contexts are implicit
and destructors do not communicate.

`maps[axis][child_coordinate]` is `Some(parent_coordinate)` or `None` to drop that
coordinate, matching source -1. This is coordinate remapping, not axis reordering.
Both sides have the same order; each map length matches its child extent, and
retained parent coordinates must be in range. Inactive ranks still supply the
parent-order map list. Actual scalar tensors are representable with empty maps;
`None` replaces upstream scalar dummy tensors used for inactivity.

Gather reads only locally owned child requests through the parent communicator,
then updates the child through its explicit write collective. Retained coordinate
maps must be injective: the source inverse-map construction silently overwrites
duplicate images, which cannot define a reliable gather. Scatter may contain
collisions; they accumulate through the parent write. Dense source zeros are not
scattered, following `read_local_nnz`, so skipped and zero-only positions do not
apply beta to the destination. No global tensor gather is used by either path.

## Indexed-write coefficient order

The source dense write leaf (`sparse_rw.cxx:795-839`) performs
`beta*old + alpha*first_input`, then prepends later duplicate contributions as
`alpha*input + current`. Rust dense and compressed `write_scaled` now preserve
that order, including left-sided multiplication for noncommutative algebras.
This corrects the previous Rust right-sided implementation. It is distinct from
`algstrct::acc` used for subworld accumulation, whose right-sided coefficients
are intentionally unchanged. Sparse pair write semantics are likewise unchanged.

## Packed and sparse combinations

All source-supported storage combinations now have direct Rust methods:
gather can read a dense, packed or sparse parent into a dense or packed child;
scatter can read any of those three child storage types into any of the three
parent types. Mixed variants use explicit `_dense`, `_symmetric` or `_sparse`
method suffixes. Gather into sparse storage remains unavailable, matching the
source's explicit assertion, rather than silently densifying a sparse target.

Packed gather enumerates the full logical child domain using a cyclic request
layout, without allocating an unpacked value tensor. The parent read handles
source symmetry signs, and the child write canonicalizes and accumulates signed
orbit duplicates. This is **not** a normalized slice copy: retained orbit entries
can multiply a canonical value. Beta is still applied once per touched canonical
key. Packed scatter sends only canonical nonzero source entries, not full orbits.
Sparse scatter sends all stored pairs including explicit zeros; an explicit zero
therefore touches and scales the destination, unlike an omitted dense zero.

The original `permute_multiworld` driver was run unmodified at n3/one rank:
NS passed, then its SY expected-copy assertion failed in the C++ baseline.
See `upstream-known-failures.md`. The driver row is not declared fully passing.
Dedicated exact source-semantic fixtures cover the compressed/mixed paths; no
normalization was invented to satisfy a different copy contract.
