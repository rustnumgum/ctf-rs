# Explicit all-rank extraction

`Tensor::all_pairs(nonzeros_only)` and `all_data()` return sorted global keys
and corresponding values, replicated on every rank of the supplied context.
These are explicit data-export collectives, not distributed compute kernels.
They use MPI_Allgather for byte counts followed by MPI_Allgatherv and key sorting,
following `untyped_tensor.cxx:2443-2557` at the pinned revision.

`SparseTensor` exposes the same methods. With `nonzeros_only=true`, all stored
pairs are returned, including explicit zeros. With false, implicit zeros are
materialized by the existing distributed dense conversion. `all_data()` returns
that full dense sequence. This distinction follows `read_local_nnz:1880-1894`.

`SymmetricTensor::all_pairs(nonzeros_only, unpack_sym)` returns packed canonical
pairs unless `nonzeros_only=false` and `unpack_sym=true`. That combination uses
distributed unpack before extraction, including antisymmetric signs and hollow
diagonal zeros. As in the source, nonzero-only reads ignore `unpack_sym`.
`all_data(unpack_sym)` extracts the corresponding full or packed value sequence.

Padding and duplicate replica layers are excluded. Empty global domains return
empty vectors. MPI byte counts/displacements must fit native signed 32-bit
counts; this interface does not invent a chunked fallback.
