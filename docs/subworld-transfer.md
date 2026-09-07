# Explicit parent/child accumulation

All ranks of the parent tensor's context call each operation. Ranks in the
selected child context pass `Some(&mut child_tensor)` or `Some(&child_tensor)`;
other parent ranks pass `None`. Each call describes one child communicator,
whose ranks may be reordered or noncontiguous within the parent.

```rust,ignore
parent.add_to_subworld(child.as_mut(), &child_distribution, alpha, beta);
parent.add_from_subworld(child.as_ref(), &child_distribution, alpha, beta);
```

The explicit child distribution must be the same on all parent ranks and match
the actual tensors on participating ranks. Shapes are unchanged. The formulas
are `child = parent*alpha + child*beta` and
`parent = child*alpha + parent*beta`, preserving pinned `algstrct::acc` order
also for noncommutative algebras. Source tensors are unchanged.

Canonical source owners send entries through the parent communicator directly
to all destination owners, including replicas. No root assembles the tensor.
The implementation derives matching offset streams from the distributions and
the child-rank-to-parent-rank orientation. Only values cross the communicator;
keys are not carried per element. Each side traverses its own local coordinates
in global-key order. Native source buffer reuse and closed-form bucket-count
optimizations remain separate backlog items.

Child tensors borrow their child context. Finish transfers and drop those
tensors before explicitly closing the child context. Destructors do not perform
collective communication.

## Compressed tensors

`SymmetricTensor` provides the same operations with an explicit
`SymmetricDistribution`, requiring equal shape and symmetry links. Only canonical
compressed entries are routed; no rectangular unpack or root gather occurs.
The receive side writes packed offsets directly, preserving padding/structural
holes and replica ownership. Ring coefficients remain right multipliers:
incoming*alpha + old*beta. Inactive parent ranks pass None even for reversed or
noncontiguous child memberships. Source values remain unchanged.

This path uses the same value-only protocol as dense subworld transfer, with
symmetry-bounded traversal and packed offsets. It does not claim sparse-pair
storage compatibility with the pinned dense-buffer routine. Rank orientation is
applied directly to peer buckets rather than creating source mirror buffers and
then forwarding them in a second point-to-point phase.
