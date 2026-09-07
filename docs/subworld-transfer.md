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
The current implementation uses explicit serialized key routing; optimized
upstream cyclic-reshuffle buffer kernels remain a separate backlog item.

Child tensors borrow their child context. Finish transfers and drop those
tensors before explicitly closing the child context. Destructors do not perform
collective communication.
