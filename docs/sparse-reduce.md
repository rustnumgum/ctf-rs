# CSR/CCSR partitioned reduction

`Csr::reduce(context, root, algebra)` and `Ccsr::reduce` return `Some(matrix)`
only at the requested root. Inputs remain owned by the caller. The operation
is collective and takes an explicit communicator; no communicator is retained
in the matrix and no communication occurs in ordinary Drop.

The implementation ports pinned `tensor/algstrct.cxx::csr_reduce`:

1. CSR chooses the smallest divisor s>=2 of process count p; CCSR uses s=p.
2. Split contiguous groups of s ranks and corresponding column groups.
3. Cyclically partition each matrix's rows into s pieces and exchange pieces
   all-to-all within each contiguous group.
4. Merge received pieces with the source rank-ordered binary tree.
5. Recursively reduce each row partition in its column communicator.
6. In the requested root's contiguous group, Gather/Gatherv only the disjoint
   reduced row partitions; assemble them in partition order at root.
7. Explicitly close both temporary communicators.

This is not an initial whole-matrix gather followed by serial addition. Final
root assembly is intrinsic to the source reduce API. Rust wire payloads carry
shape and stored COO entries rather than C++ object/packed-buffer ABI. CCSR
transport remains proportional to stored entries, not its logical row count.
Stored zeros and noncommutative monoid merge order are retained.

The primitive is now used by the explicit CSR/CCSR replicated/virtual output
layer described in sparse-replicate.md. Integration into all automatically
assembled sparse moving-output plans remains unfinished; existing
two-dimensional sparse GEMM paths are not relabeled as that full stack.
