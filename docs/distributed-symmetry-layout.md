# Pinned distributed compressed layout

Source: cc4s CTF f69cbb46e23bc2f39cda5722ce096f56301dab4f.
This is an implementation guide from source inspection, not runtime validation.

## Allocation and indexing

For axis d, let physical phase be H[d], virtual phase V[d], and total phase
P[d]=H[d]*V[d]. Normal map_symtsr equalizes total phases within each symmetry
group, but permits different physical topology axes (mapping/mapping.cxx
476-545). Padding is (-n[d]) mod P[d], giving local extent
L[d]=(n[d]+padding[d])/P[d]. Each virtual block allocates
sy_packed_size(L,sym) slots (tensor/untyped_tensor.cxx 640-670).

Crucially, sy_packed_size uses combinations with repetition for SY, AS and SH
alike (shared/util.cxx 10-35). AS/SH's strict canonical count is not the dense
physical allocation size.

Global coordinates are first sorted within each symmetry group. AS attaches
permutation parity; AS/SH repeated coordinates are structural zeros
(redistribution/sparse_rw.cxx 962-1080). Canonical coordinates determine:

- Physical digit: x[d] mod H[d].
- Virtual digit: floor(x[d]/H[d]) mod V[d], axis 0 fastest.
- Local quotient: q[d]=floor(x[d]/P[d]).
- Offset: virtual_block * packed_block_size + local_SY_packed_rank(q).

For one group of m axes, local_SY_packed_rank is
sum over d=0..m-1 of binomial(q[d]+d,d+1). Equal total phases preserve the
ordering of q for globally canonical x. Physical mapping chains compose rank
digits as r0+H0*rchild (mapping/mapping.cxx 74-91).
Key bucketing and packed indexing are in redistribution/sparse_rw.cxx
179-275,449-887.

## Holes are allocated, not semantic replicas

For n=5 and SY on a 2x2 process grid, P=H=(2,2), L=(3,3), six slots are
allocated per rank. Rank coordinates (1,0) contain:

| Offset | Local q | Raw global coordinates | Validity |
|---|---|---|---|
| 0 | (0,0) | (1,0) | noncanonical hole |
| 1 | (0,1) | (1,2) | canonical |
| 2 | (1,1) | (3,2) | noncanonical hole |
| 3 | (0,2) | (1,4) | canonical |
| 4 | (1,2) | (3,4) | canonical |
| 5 | (2,2) | (5,4) | padded hole |

There are 24 allocated slots, but only 15 canonical global entries. There is
no triangular rank restriction or symmetry-specific rank permutation.
zero_padding checks global canonicality (strict for AS/SH), zeroing invalid
slots (redistribution/pad.cxx 457-655). Local reads depad and filter generated
keys (sparse_rw.cxx 1252-1325; pad.cxx 51-213).

## Operations and Rust consequences

- get_local_pairs delegates to read_local/read_local_nnz (interface/tensor.cxx
  384-405). unpack_sym=true materializes NS storage through summation
  (tensor/untyped_tensor.cxx 2371-2395).
- Dense-to-sparse conversion filters these same holes (untyped_tensor.cxx
  1745-1795).
- SY-to-AS/SH repacking keeps SY-sized physical storage and clears invalid
  slots; changing NS links uses summation (untyped_tensor.cxx 223-270).
- Cyclic reshuffling transfers globally canonical coordinates only and clears
  destination storage first (redistribution/cyclic_reshuffle.cxx 261-280,
  632-643,737-739).

The existing Rust rectangular Distribution offsets cannot be used unchanged.
Distributed compressed offsets need SY-like local packed rank for all three
symmetry types, plus separate global validity and AS parity. Do not substitute
globally packed-linear ownership or the local AS strict packed rank. The normal
supported mapping path must enforce equal total phases within groups; upstream
custom mapping construction bypasses map_symtsr (untyped_tensor.cxx 488-543).
