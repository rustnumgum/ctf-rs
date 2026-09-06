# Source provenance

Reference repository: https://gitlab.cc4s.org/cc4s/ctf
Reference commit: f69cbb46e23bc2f39cda5722ce096f56301dab4f

`LICENSE` preserves the upstream notice and terms for adapted material. It is
not a declaration that Edgar Solomonik authored the new Rust implementation.
Do not stamp independently written files with the upstream author's copyright.

* `src/mapping.rs`: map-chain phase/rank and topology reorder/inverse routines
  adapted from `src/mapping/{mapping,topology,distribution}.cxx`.
* `src/map_tensor.rs`: physical-axis assignment and symmetry-phase coordination
  adapted from `map_tensor`/`map_symtsr` in `src/mapping/mapping.cxx`.
* `src/topology_candidates.rs`: ordered factorization, adjacent folding and
  permutation/folding enumeration adapted from `src/mapping/topology.cxx`.
* `src/node_aware.rs`: inter-node grid factor assignment/tree enumeration adapted
  from `src/mapping/node_aware_dist.cxx`, whose author notice names Andreas Irmler;
  its corresponding upstream header carries the 2022 Edgar Solomonik notice.
* `src/tensor.rs`: Rust distributed storage implementation using the upstream
  cyclic distribution and key-bucket exchange approach; not a literal complete
  port of all optimized redistribution kernels.
  Dense slice extraction follows `redistribution/slice.cxx`'s local extraction
  and physical cyclic rank-shift communication; Rust reindexes virtual blocks
  explicitly rather than relying on raw byte copies.
* `src/algebra.rs`, `src/context.rs`, `src/ffi/mpi.rs`: new Rust traits, lifetimes,
  serialization and native-call encapsulation. Upstream responsibility references
  document compatibility targets, not authorship of these files.
* `src/summation.rs`: local NS sequential summation control flow and alpha/beta
  ordering adapted from `src/summation/sym_seq_sum.cxx`; Rust index offset tables
  replace byte offsets and explicit C++ buffer management.
  Its virtual-block and replicated-block layers follow `tsum_virt::run` and
  `tsum_replicate::run` in `src/summation/sum_tsr.cxx`.
* `src/contraction.rs`: dense NS reference contraction and custom function/alpha
  ordering adapted from `src/contraction/sym_seq_ctr.cxx`.
  Virtual traversal follows `ctr_virt::run` in `ctr_tsr.cxx`; replicated execution
  follows `ctr_replicate::run` in `ctr_comm.cxx`, including root reduction and
  input-replica clearing.
  Folded CPU batch layout follows `interface/semiring.cxx::gemm_batch`; output
  operand swapping and prescaling follow `sym_seq_ctr_inr`.
* `src/ctr_2d.rs`: panel packing/broadcast, output reduction/scatter and layer
  propagation adapted from `ctr_2d_general::run` and `find_bsizes` in
  `src/contraction/ctr_2d_general.cxx`.
* `src/diagonal.rs`: repeated-coordinate key projection/insertion follows the
  index deletion/insertion rules in `tensor::extract_diag`. Current dense data
  transfer uses canonical-key redistribution, not the optimized upstream dense
  mapped-summation path; do not claim exact communication parity for this path.
* `tests/upstream_dense.rs`: numerical identities from `test/diag_ctr.cxx` and
  `test/reduce_bcast.cxx`, retaining original residual metrics and tolerances.
* `src/linalg.rs`, `src/ffi/linalg.rs`: new local-kernel interface and bindings to
  standard BLAS/LAPACK operations. No C++ CTF wrapper is linked or copied here.

Keep attribution with actual adaptations and add per-source provenance as the
port grows. The source/test inventory is a scope ledger, not a claim that all
listed source files have been ported.
