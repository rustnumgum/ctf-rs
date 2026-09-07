# Native COO CPU multiplication

`Coo::default_coomm` ports default_coomm: scale each output as old*beta,
then visit stored entries and columns, accumulating alpha*(A*B) with +=.
`Coo::coomm` ports the distinct generic semiring fallback: beta must be one,
and addition is contribution-first. Neither method converts to CSR or sorts
COO entries. Duplicate entries and explicitly stored zeros remain observable.

`Coo::coomm_kernel` ports Kernel<f,g> with independent A/B/C types and a
caller-supplied accumulator g(f(a,b),c). Alpha must be absent or one, beta
must be one. This does not claim that ordinary Bivar_Function COO works:
that source class does not override ccoomm and reaches the base assertion.

`sparse_2d::execute_coo_dense` connects the native COO leaf to explicit
recursive 2D communication. Sparse broadcasts preserve entry order using
per-block payload sizes; dense B has fixed blocks, and moving dense output
uses cyclic reduction followed by new + beta*old. Sparse operands are never
gathered or converted into dense inputs.

The default CPU Rust loop replaces any vendor-specialized COO implementation;
it retains the source computation and does not depend on MKL sparse symbols.
Automatic sparse plan assembly/COO-versus-CSR selection remains unfinished.
