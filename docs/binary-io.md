# Distributed binary tensor I/O

`write_dense_to_file(path, offset)` and `read_dense_from_file(path, offset)`
are explicit collectives on the tensor context, for dense, sparse and compressed
storage. Offset is measured in bytes. Callers supply shape, algebra and symmetry;
the file contains only values in increasing global-key order, without metadata.

The pinned `untyped_tensor.cxx:3819-3909` assigns consecutive chunks of size
`N / np + (rank < N % np)` and start `rank * (N / np) + min(rank, N % np)`.
Each rank obtains its own chunk through indexed distributed reads, then performs
independent-offset MPI-IO between collective file open and explicit close. Reads
redistribute the chunk back to the tensor's layout. No all-data gather is used.

Dense reads overwrite existing values. Sparse reads replace storage and discard
zero entries, following source sparsification. Compressed writes expand logical
orbits; reads select canonical entries without summing noncanonical permutations.
Writes create a missing file but do not truncate an existing suffix. A short read
is an execution error, not an implicit zero fill.

Values use the algebra element's `Wire` encoding. Built-in integer, real and
complex encodings match native scalar bytes on the supported little-endian WSL
and Windows targets; custom elements define their own encoding, not a C++ ABI.
The explicit offset is honored for every storage representation. This corrects
the pinned sparse/symmetric recursion's accidental omission of its offset.
The Rust APIs work for arbitrary tensor order instead of reproducing the source
unpack branch's hard-coded two-index string.
