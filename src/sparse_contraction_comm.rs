// Adapted from cc4s CTF contraction/{spctr_comm,spctr_tsr}.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Replicated sparse-output contraction over explicit communicator fibers.
//!
//! Output communicators must be ordered orthogonal fibers. After each reduction
//! only that fiber's root participates in the next one, so the result is present
//! only at the intersection root. Received sparse input vectors are cleared on
//! nonroots; received dense input storage retains its shape and is zeroed.

use crate::{
    algebra::{Semiring, Wire},
    context::Context,
    sparse_formats::{Ccsr, Coo, Csr},
};

fn encode_coordinates<T: Wire + Clone>(blocks: &[Coo<T>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    u64::try_from(blocks.len()).unwrap().encode(&mut bytes);
    for block in blocks {
        let (rows, columns) = block.shape();
        u64::try_from(rows).unwrap().encode(&mut bytes);
        u64::try_from(columns).unwrap().encode(&mut bytes);
        u64::try_from(block.entries().len()).unwrap().encode(&mut bytes);
        for (row, column, value) in block.entries() {
            u64::try_from(*row).unwrap().encode(&mut bytes);
            u64::try_from(*column).unwrap().encode(&mut bytes);
            value.encode(&mut bytes);
        }
    }
    bytes
}

fn decode_coordinates<T: Wire + Clone>(bytes: &[u8]) -> Vec<Coo<T>> {
    assert!(bytes.len() >= 8);
    let count = usize::try_from(u64::decode(&bytes[..8])).unwrap();
    let mut position = 8;
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        assert!(position + 24 <= bytes.len());
        let rows = usize::try_from(u64::decode(&bytes[position..position + 8])).unwrap();
        let columns = usize::try_from(u64::decode(&bytes[position + 8..position + 16])).unwrap();
        let entries = usize::try_from(u64::decode(&bytes[position + 16..position + 24])).unwrap();
        position += 24;
        let width = 16 + T::WIDTH;
        let end = position + entries * width;
        assert!(end <= bytes.len());
        let entries = bytes[position..end]
            .chunks_exact(width)
            .map(|entry| {
                (
                    usize::try_from(u64::decode(&entry[..8])).unwrap(),
                    usize::try_from(u64::decode(&entry[8..16])).unwrap(),
                    T::decode(&entry[16..]),
                )
            })
            .collect();
        blocks.push(Coo::new(rows, columns, entries));
        position = end;
    }
    assert_eq!(position, bytes.len());
    blocks
}

fn broadcast_coordinates<T: Wire + Clone>(context: &Context<'_>, blocks: &mut Vec<Coo<T>>) {
    let mut bytes = if context.rank() == 0 {
        encode_coordinates(blocks)
    } else {
        Vec::new()
    };
    let mut length = [u64::try_from(bytes.len()).unwrap()];
    context.broadcast(0, &mut length);
    if context.rank() != 0 {
        bytes.resize(usize::try_from(length[0]).unwrap(), 0);
    }
    context.inner.broadcast(0, &mut bytes);
    if context.rank() != 0 {
        *blocks = decode_coordinates(&bytes);
    }
}

fn broadcast_dense<T: Wire + Clone>(context: &Context<'_>, blocks: &mut [Vec<T>]) {
    let mut bytes = Vec::with_capacity(blocks.iter().map(Vec::len).sum::<usize>() * T::WIDTH);
    if context.rank() == 0 {
        for value in blocks.iter().flatten() {
            value.encode(&mut bytes);
        }
    } else {
        bytes.resize(blocks.iter().map(Vec::len).sum::<usize>() * T::WIDTH, 0);
    }
    context.inner.broadcast(0, &mut bytes);
    if context.rank() != 0 {
        let mut values = bytes.chunks_exact(T::WIDTH);
        for value in blocks.iter_mut().flatten() {
            *value = T::decode(values.next().unwrap());
        }
    }
}

fn block_count(dimensions: &[usize], indices: &[usize]) -> usize {
    indices.iter().map(|&index| dimensions[index]).product()
}

/// Replicate sparse inputs, execute CSR virtual blocks, and reduce sparse C.
/// `c_comms` must be ordered orthogonal fibers; `Some` is returned only at
/// their intersection root. Nonroot sparse input replicas are cleared.
pub fn replicated_csr<A: Semiring>(
    algebra: &A,
    a_comms: &[&Context<'_>],
    b_comms: &[&Context<'_>],
    c_comms: &[&Context<'_>],
    virtual_dimensions: &[usize],
    indices: [&[usize]; 3],
    a: &mut Vec<Csr<A::Element>>,
    b: &mut Vec<Csr<A::Element>>,
    mut c: Vec<Csr<A::Element>>,
    alpha: &A::Element,
    beta: &A::Element,
) -> Option<Vec<Csr<A::Element>>>
where
    A::Element: Wire,
{
    assert_eq!(a.len(), block_count(virtual_dimensions, indices[0]));
    assert_eq!(b.len(), block_count(virtual_dimensions, indices[1]));
    assert_eq!(c.len(), block_count(virtual_dimensions, indices[2]));
    for communicator in a_comms {
        let mut blocks: Vec<_> = a.iter().map(Csr::to_coo).collect();
        broadcast_coordinates(communicator, &mut blocks);
        *a = blocks.into_iter().map(|block| block.to_csr()).collect();
    }
    for communicator in b_comms {
        let mut blocks: Vec<_> = b.iter().map(Csr::to_coo).collect();
        broadcast_coordinates(communicator, &mut blocks);
        *b = blocks.into_iter().map(|block| block.to_csr()).collect();
    }

    let output_root = c_comms.iter().all(|communicator| communicator.rank() == 0);
    if !output_root {
        c = c.into_iter().map(|block| {
            let (rows, columns) = block.shape();
            Coo::new(rows, columns, Vec::new()).to_csr()
        }).collect();
    }
    let child_beta = if output_root { beta.clone() } else { algebra.zero() };
    let one = algebra.one();
    crate::sparse_virtual::execute(
        virtual_dimensions,
        indices,
        &child_beta,
        &one,
        |blocks, leaf_beta| {
            c[blocks[2]] = a[blocks[0]].multiply_sparse(
                &b[blocks[1]],
                alpha,
                leaf_beta,
                Some(&c[blocks[2]]),
                algebra,
            );
        },
    );

    let mut result = Some(c);
    for communicator in c_comms {
        let Some(blocks) = result.take() else { break };
        let root = communicator.rank() == 0;
        let mut reduced = Vec::with_capacity(if root { blocks.len() } else { 0 });
        for block in blocks {
            let block = block.reduce(communicator, 0, algebra);
            if root {
                reduced.push(block.unwrap());
            }
        }
        result = root.then_some(reduced);
    }
    if a_comms.iter().any(|communicator| communicator.rank() != 0) {
        a.clear();
    }
    if b_comms.iter().any(|communicator| communicator.rank() != 0) {
        b.clear();
    }
    result
}

/// Replicate CCSR A and fixed-size dense B blocks, execute virtual blocks, and
/// reduce sparse C. `c_comms` must be ordered orthogonal fibers; `Some` is
/// returned only at their intersection root. Nonroot sparse A is cleared and
/// nonroot dense B replicas are zeroed without changing their block shapes.
pub fn replicated_ccsr_dense<A: Semiring>(
    algebra: &A,
    a_comms: &[&Context<'_>],
    b_comms: &[&Context<'_>],
    c_comms: &[&Context<'_>],
    virtual_dimensions: &[usize],
    indices: [&[usize]; 3],
    a: &mut Vec<Ccsr<A::Element>>,
    b: &mut Vec<Vec<A::Element>>,
    mut c: Vec<Ccsr<A::Element>>,
    alpha: &A::Element,
    beta: &A::Element,
) -> Option<Vec<Ccsr<A::Element>>>
where
    A::Element: Wire,
{
    assert_eq!(a.len(), block_count(virtual_dimensions, indices[0]));
    assert_eq!(b.len(), block_count(virtual_dimensions, indices[1]));
    assert_eq!(c.len(), block_count(virtual_dimensions, indices[2]));
    for communicator in a_comms {
        let mut blocks: Vec<_> = a.iter().map(Ccsr::to_coo).collect();
        broadcast_coordinates(communicator, &mut blocks);
        *a = blocks.into_iter().map(|block| block.to_ccsr()).collect();
    }
    for communicator in b_comms {
        broadcast_dense(communicator, b);
    }

    let output_root = c_comms.iter().all(|communicator| communicator.rank() == 0);
    if !output_root {
        c = c.into_iter().map(|block| {
            let (rows, columns) = block.shape();
            Coo::new(rows, columns, Vec::new()).to_ccsr()
        }).collect();
    }
    let child_beta = if output_root { beta.clone() } else { algebra.zero() };
    let one = algebra.one();
    crate::sparse_virtual::execute(
        virtual_dimensions,
        indices,
        &child_beta,
        &one,
        |blocks, leaf_beta| {
            let columns = c[blocks[2]].shape().1;
            c[blocks[2]] = a[blocks[0]].multiply_dense(
                columns,
                &b[blocks[1]],
                alpha,
                leaf_beta,
                Some(&c[blocks[2]]),
                algebra,
            );
        },
    );

    let mut result = Some(c);
    for communicator in c_comms {
        let Some(blocks) = result.take() else { break };
        let root = communicator.rank() == 0;
        let mut reduced = Vec::with_capacity(if root { blocks.len() } else { 0 });
        for block in blocks {
            let block = block.reduce(communicator, 0, algebra);
            if root {
                reduced.push(block.unwrap());
            }
        }
        result = root.then_some(reduced);
    }
    if a_comms.iter().any(|communicator| communicator.rank() != 0) {
        a.clear();
    }
    if b_comms.iter().any(|communicator| communicator.rank() != 0) {
        for block in b.iter_mut() {
            block.fill(algebra.zero());
        }
    }
    result
}
