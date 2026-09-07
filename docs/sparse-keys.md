# Raw sparse key pinning

`sparse_keys` ports ConstPairIterator::pin and depin from the pinned
tensor/algstrct.cxx. `KeyMetadata` holds logical lengths, total phases,
virtual dimensions and physical ranks. Input/output vectors retain separate
virtual blocks with dimension zero varying fastest.

Pinning divides each global coordinate by its full physical-plus-virtual
phase and encodes the result with ceil(length/phase) local radices. Depinning
reconstructs each coordinate from the local coordinate and the virtual
residue `(virtual_coordinate * physical_phase + physical_rank)`. The source
padding predicate selects whether reconstructed out-of-shape entries must
be removed. Values, explicit zeros and relative pair ordering are preserved;
empty blocks stay present. Neither transform communicates or creates CSR.

The mapped sparse-A/dense-B/dense-C production path now groups exchanged
global keys by target virtual block, sorts within each block, then calls
pin_blocks before its replication and virtual leaf execution. This replaces
the previous flattened-offset/modulo conversion. Replication sends all block
counts followed by a concatenated key/value payload once per fiber.

The source spctr_pin_keys wrapper leaves nX null for operand C before calling
pin. That invalid pointer orchestration is not reproduced: Rust provides
the defined output depin operation directly. Automatic sparse-output plan
assembly still needs to connect it to the full source execution pipeline.

`sparse_cost::KeyPinning` retains the source's output double-pass time and
its memory switch fallthrough: A counts A+B+C, B counts B+C, C counts C.
This is a source estimate, not Rust allocation accounting.
