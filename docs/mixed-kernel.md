# Mixed-type CPU Kernel functions

`kernel` ports the dense GEMM, CSR/dense, CSR/CSR-to-dense and strided xpy
loops from interface/kernel.h Bivar_Kernel and Monoid_Kernel. A, B and C
may have independent element types. Each stored traversal step invokes
g(f(a,b),c); neither f nor g is silently replaced by a semiring operation.
Dense transpose selection preserves source non-N behavior without conjugation.

These native CPU loops use existing initialized output objects and apply no
implicit alpha/beta coefficients. Sparse structural absences are skipped;
explicit zeros and duplicate coordinates remain evaluations.

`kernel::csr_sparse` ports the sparse-output symbolic/numeric row algorithm:
the first reachable product directly initializes an output value, subsequent
products invoke g, and sorted product structure is merged into old C through
g(product,old). Cancellations remain stored. This retains source last-copy
behavior for duplicate old C coordinates without imposing an additive zero
or Default bound on the result type.

`sparse_2d::execute_coo_dense` accepts independent A/B/output types and
communicates each with its Wire representation. The output algebra governs
reductions and beta; the child applies the source leaf coefficient rules.
Automatic distributed mixed-type plan dispatch is still pending. COO Kernel
behavior is documented in sparse-coo.md.
