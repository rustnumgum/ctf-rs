# Explicit distributed schedules

`Schedule<A>` records one-output operations with caller-assigned tensor IDs,
input IDs, an estimated duration in seconds, and a Rust closure. The closure
receives a mutable ordered map of its child tensors, and invokes the existing
direct tensor APIs. Input IDs include the output for read-modify-write. All
parent ranks record the same metadata and execute collectively; no global
recorder, expression-template AST, C++ operator syntax or backend plugins exist.
Costs come from callers/the existing planners; automatic extraction from an
arbitrary Rust closure is not provided.

The dependency graph preserves read-after-write, write-after-read and
write-after-write ordering. Execution resets dependency counts, allowing replay.
Each wave sorts ready operations by decreasing cost, chooses the longest
contiguous window satisfying the upstream minimum-cost imbalance criterion,
then assigns ranks by midpoint sampling of equal-cost processor blocks.
`max_partitions` optionally limits concurrent operations. Equal costs retain
ready-queue order.

Execution follows the source's communication-down / compute / communication-up
phases. Every parent rank participates in explicit tensor transfer; each selected
task runs only on its own child communicator. Child tensors use cyclic layouts,
parent tensors retain their distributions, and only each declared output is
transferred back. Child tensors are dropped before explicitly closing the child
context. No destructor communicates, and no root gathers full tensor data.
The six returned timing fields retain source wall/accumulated-imbalance meanings.

## Necessary corrections to the pinned implementation

* Keep cost sums and rank sampling in f64. Source `max_cost` and
  `color_sample_point` are int, collapsing ordinary subsecond estimates to zero.
* Zero-dependency operations enter the ready queue. Source operations without
  inputs never get a dummy-root dependency and are not enqueued.
* Include prior-writer edges for overwrites, not only prior readers. Otherwise
  repeated assignment can execute concurrently or backwards.
* Allocate tensors on the actual child context and pass `None` on nonmembers.
  Source clones the indexed wrapper without remapping the world and dereferences
  `local_clone` even in its null branch.

Dense semiring operations are supported by this execution path. It does not
claim sparse/compressed subworld execution or automatic cost-model attachment;
the optimized cyclic-reshuffle implementation remains a separate backlog item.

Source: `interface/schedule.{h,cxx}` at the pinned revision. Pure graph/partition
tests and the `schedule` MPI test cover the core production path, including
independent operations on proper subcommunicators, exact integer sums and
contractions, all dependency hazards, no-input roots and replay.
