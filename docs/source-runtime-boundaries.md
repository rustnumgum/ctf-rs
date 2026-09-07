# Declared interfaces without coherent pinned runtime semantics

These findings concern commit f69cbb46e23bc2f39cda5722ce096f56301dab4f, not
mathematical definitions or later upstream versions. They prevent counting a
declaration as supported computational capability.

## Boolean norms

Boolean `norm_infty` is a coherent same-type MAXABS reduction:
`interface/tensor.h:1180-1183`, `tensor.cxx:1305-1311`, and the bool ordering/abs
in `set.h:313-317,437-447`. The double overload converts the boolean result
(`tensor.cxx:1544-1550`). Rust dense/sparse implementations return exactly 1.0
when a logical true exists, otherwise 0.0, including empty tensors.

The bool norm1 path in `tensor.cxx:1337-1343` directly converts a bool indexed
term to double. `term.cxx:328-334` constructs a double scalar and accumulates the
term; `idx_tensor.cxx:232-240,295-309` builds an ordinary non-custom summation.
That leaf uses the destination double algebra on the source's bool-sized pointer
(`sym_seq_sum.cxx:385-392`). This is not a defined count or boolean-OR conversion.
The commented cast alternative is not a usable bool cast-to-double implementation
(`set.h:968-970` only supplies bool cast-to-int). Rust intentionally provides no
bool norm1 method instead of inventing numerical semantics for invalid reads.
Compressed bool tensors also remain unavailable under the current Group-bound
storage API; this finding does not close that distinct semantic-design backlog.

## Sparse subworld accumulation

`untyped_tensor.cxx:1241-1313` always calls dense `cyclic_reshuffle` for
add_to/from_subworld. Its buffer sizes and accesses use dense element arrays
(`cyclic_reshuffle.cxx:477-523`); `orient_subworld` similarly sends
`odst->size * el_size` from raw tensor data (`untyped_tensor.cxx:1043-1060`).
Neither sparse key/value records nor nnz counts are handled. The absence of a
sparse assertion does not make this a coherent sparse implementation. Supporting
a new sparse subworld API would require a separately defined pair-based algorithm,
not direct migration of a working path. Compressed dense subworld reshuffle is
separate and remains in scope.
