# Sparse coordinate text I/O

`Tensor`, `SparseTensor` and `SymmetricTensor` provide `read_sparse_from_file` and
`write_sparse_to_file` for f32, f64, i32 and i64. Both operations are collective
on the tensor's explicit context, including subcontexts.

Records are zero-based coordinates followed optionally by a value, separated
by whitespace and terminated by a newline. `reverse_order` reverses coordinate
order; `with_values=false` omits values when writing and reads each record as
the multiplicative identity. Reads add records to existing data and combine
duplicate keys. Writers emit canonical pairs in rank/local order: dense and
compressed storage omit zeros, while sparse storage retains explicitly stored
zeros, matching the pinned `read_local_nnz` behavior.
Real values use the source's fixed six fractional digits, not lossless encoding.
Compressed tensors write canonical packed entries, not expanded symmetry orbits.
Symmetric reads canonicalize input permutations and their signs. Shape and
symmetry metadata are supplied by the caller, not stored in this text format.

MPI-IO reads retain the pinned 300-byte overlap and newline ownership rule.
Small reads are clamped at EOF and empty files are supported. A boundary record
without a newline in the overlap, or an unterminated final record, is rejected;
there are no retry scans or serial fallback. Writes replace existing contents
and use MPI scan-derived offsets for collective file writes.

The Rust implementation intentionally corrects two unsafe source details:
native calls use the supplied communicator rather than MPI_COMM_WORLD, and
values are parsed into their actual type instead of scanning non-double formats
into a double pointer. It does not use C++ code or ABI shims.
