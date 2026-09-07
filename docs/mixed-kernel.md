# Mixed-type CPU Kernel functions

`kernel` ports the dense GEMM, CSR/dense, CSR/CSR-to-dense and strided xpy
loops from interface/kernel.h Bivar_Kernel and Monoid_Kernel. A, B and C
may have independent element types. Each stored traversal step invokes
g(f(a,b),c); neither f nor g is silently replaced by a semiring operation.
Dense transpose selection preserves source non-N behavior without conjugation.

These native CPU loops use existing initialized output objects and apply no
implicit alpha/beta coefficients. Sparse structural absences are skipped;
explicit zeros and duplicate coordinates remain evaluations. Sparse output
Kernel functions and automatic distributed mixed-type dispatch are not yet
claimed complete. COO Kernel behavior is documented in sparse-coo.md.
