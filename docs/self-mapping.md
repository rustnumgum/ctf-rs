# Repeated-index self mapping

`self_mapping` ports mapping.cxx map_self_indices and check_self_mapping.
Repeated labels add adjacency to a copy of the supplied symmetry table. Each
previously unmapped earlier occurrence becomes a unit virtual map. If any
such map was created, the source deliberately skips symmetry coordination
on that call; a subsequent source pass may coordinate phases. This behavior
is retained rather than replaced by an automatic fixpoint/retry.

The checker walks repeated indices backwards, requires equal phases and
childless maps for each repeated set, and forbids a physical map on the
earlier occurrence. Its physical-chain check compares every descendant to
the current head axis plus one, not merely adjacent axis pairs. Source
top-level scanning stops at virtual map heads; these rules are not silently
strengthened into another planner's invariants.

The existing contraction mapping preflight now calls this source self-check
instead of maintaining a separate unique-label approximation. Automatic
repeated-index candidate generation still needs to use map_self_indices;
the current tensor diagonal-extraction paths are not claimed to complete
that planner integration.

`mapping::calc_dim` additionally ports distribution.cxx's explicit size and
edge metadata walk. Physical factors divide both block and virtual edge
lengths; virtual factors divide virtual edges and virtual storage size.
All divisions truncate in chain order, as upstream; this is not the logical
shape ceiling rule used by Distribution::block_shape.
