# Dense value-only cyclic reshuffle

`Tensor::redistribute` now transmits values only; global keys are not sent with
each element. Sender and receiver derive the same ordered per-peer stream from
the old and new distributions. This follows the NS global-order traversal,
pack / Alltoallv / unpack structure in `redistribution/cyclic_reshuffle.cxx`.

The local traversal starts at each mode's physical rank residue and advances
dimension zero fastest by physical phase, stopping at real lengths. Existing
local-offset rules place values into virtual blocks. It does not scan the global
tensor or gather/sort all global keys. Only canonical old owners send; every new
replica receives its values, preserving the Rust storage invariant. Padding is
excluded from messages and initialized to the algebra's identity at destination.

Plans contain local send/receive offset tables. Built-in and custom Wire elements
share this path without cloning the old element array. Communication still uses
the existing communicator-scoped variable-size exchange. There is no speedup
claim: transmitted payload shrinks by eight key bytes per transferred element,
while offset tables consume local memory.

This closes the NS value-stream path, not every optimized source reshuffle case.
Closed-form bucket counts, buffer reuse, compressed symmetry, nonzero offsets,
permutation arguments and subworld integration remain to be ported. The source's
scalar rank-zero storage convention is represented by the existing Rust explicit
replica rules instead of silently discarding other stored replicas.
