# Source orchestration still required above canonical sums

Pinned cc4s CTF f69cbb46e23bc2f39cda5722ce096f56301dab4f.
The implemented sum_canonical_from corresponds to symmetry-disabled
sum_tensors, not sym_sum_tsr. This source audit records the remaining bridge.

In summation/summation.cxx, sym_sum_tsr first validates/unfolds, handles zero
edges and self-reduces repeated indices (1170-1219), then extracts remaining
diagonals unless run_diag (1236-1249). Alignment changes B's index map and
returns an AS sign (1275-1281; symmetry/sym_indices.cxx 40-153).
overcounting_factor (1282-1301; sym_indices.cxx 402-433) cancels fully reduced
AS runs, gives factorial multiplicity to SH runs, and leaves SY to its separate
diagonal-sensitive handling. Coefficients are formed by additive operations
and negation, not casting integers into arbitrary scalar algebras.

## Broken-link recursion must precede permutation enumeration

The broken-link test (summation.cxx 1766-1835) requires shared labels to occur
adjacently, in order, inside a compatible other-operand group. Compatibility
includes parity: AS matches AS, not SH. Fully reduced SY has an additional
broken condition. The source unfolds one broken link to NS and recursively
processes further broken links; only the final broken-link stage enumerates
operation permutations (1308-1371).

The existing Rust sym_permutations::enumerate mirrors get_sym_perms, whose
normalization considers shared non-NS groups preserved. Calling it directly
on arbitrary original AS/SH operands without source unfolding is incorrect:
it may deduplicate terms which the recursion is supposed to expose. Do not
replace that recursion with an inferred cancellation shortcut or treat a
passing compatible-symmetry fixture as proof of general mixed-symmetry support.

The final permutation loop uses discovery order, signed aligned alpha times
the overcounting factor, original beta on the first task and one thereafter
(1391-1405). Raw tasks must retain only destination keys already canonical;
canonicalizing and signing rejected target keys would double count terms.

## Current executable building block

sum_canonical_from routes source canonical-owner entries to explicit output
owners/replicas, selects repeated indices, reduces input-only labels and expands
output-only labels. It intersects canonical domains without permutation sums,
and scales selected old output by beta once. It does not gather tensors or
perform implicit semantic reads. This provides the symmetry-disabled task used
by the source orchestration; optimized mapped communication and unrestricted
symmetry-aware summation remain unfinished.

## Implemented hollow branch

sum_hollow_from now executes broken-link recursion for unique-index NS/AS/SH
operands. SY and repeated-index preprocessing remain outside its explicit
contract. Each recursive source stage starts with the unadjusted coefficient
copied into new_sum before the outer stage changes alpha; it then reruns
alignment and factor handling. Only terminal canonical tasks receive the
stage-adjusted coefficient. B-side old-output scaling follows the source scaling
operation's right multiplication, distinct from raw sum beta on the left.
