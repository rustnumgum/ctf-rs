// Adapted from cc4s CTF contraction/spctr_2d_general.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! One sparse 2D contraction level. Sparse panels retain per-block variable
//! storage sizes; dense panels have the fixed block shapes supplied by the
//! contraction layout.

use crate::{
    algebra::{Semiring, Wire},
    context::Context,
    sparse_formats::{Ccsr, Coo, Csr},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layers {
    pub count: usize,
    pub index: usize,
}

#[derive(Clone, Copy)]
pub struct Panel<'c, 'r> {
    pub comm: Option<&'c Context<'r>>,
    /// Number of noncontiguous strips in one panel (ctr_lda).
    pub outer: usize,
    /// Consecutive virtual blocks per strip; zero denotes a stationary whole operand.
    pub inner: usize,
}

impl Panel<'_, '_> {
    fn size(&self) -> usize {
        self.outer * self.inner
    }

    fn positions(&self, step: usize, panels: usize) -> Vec<usize> {
        let mut positions = Vec::with_capacity(self.size());
        for strip in 0..self.outer {
            let start = (strip * panels + step) * self.inner;
            positions.extend(start..start + self.inner);
        }
        positions
    }

    fn operand_positions(&self, length: usize, step: usize, edge: usize) -> Vec<usize> {
        if let Some(comm) = self.comm {
            assert_eq!(edge % comm.size(), 0);
            let panels = edge / comm.size();
            assert_eq!(length, self.size() * panels);
            self.positions(step / comm.size(), panels)
        } else if self.inner == 0 {
            (0..length).collect()
        } else {
            assert_eq!(length, self.size() * edge);
            self.positions(step, edge)
        }
    }
}

fn encode<T: Wire + Clone>(block: &Coo<T>) -> Vec<u8> {
    let (rows, columns) = block.shape();
    let mut bytes = Vec::with_capacity(24 + block.entries().len() * (16 + T::WIDTH));
    u64::try_from(rows).unwrap().encode(&mut bytes);
    u64::try_from(columns).unwrap().encode(&mut bytes);
    u64::try_from(block.entries().len()).unwrap().encode(&mut bytes);
    for (row, column, value) in block.entries() {
        u64::try_from(*row).unwrap().encode(&mut bytes);
        u64::try_from(*column).unwrap().encode(&mut bytes);
        value.encode(&mut bytes);
    }
    bytes
}

fn decode<T: Wire + Clone>(bytes: &[u8]) -> Coo<T> {
    assert!(bytes.len() >= 24);
    let rows = usize::try_from(u64::decode(&bytes[..8])).unwrap();
    let columns = usize::try_from(u64::decode(&bytes[8..16])).unwrap();
    let count = usize::try_from(u64::decode(&bytes[16..24])).unwrap();
    let width = 16 + T::WIDTH;
    assert_eq!(bytes.len(), 24 + count * width);
    let entries = bytes[24..]
        .chunks_exact(width)
        .map(|entry| {
            (
                usize::try_from(u64::decode(&entry[..8])).unwrap(),
                usize::try_from(u64::decode(&entry[8..16])).unwrap(),
                T::decode(&entry[16..]),
            )
        })
        .collect();
    Coo::new(rows, columns, entries)
}

fn broadcast_sparse<T: Wire + Clone>(
    context: &Context<'_>,
    owner: usize,
    blocks: &mut Vec<Coo<T>>,
) {
    let encoded: Vec<Vec<u8>> = if context.rank() == owner {
        blocks.iter().map(encode).collect()
    } else {
        Vec::new()
    };
    let mut sizes = vec![0u64; blocks.len()];
    if context.rank() == owner {
        for (size, bytes) in sizes.iter_mut().zip(&encoded) {
            *size = u64::try_from(bytes.len()).unwrap();
        }
    }
    context.broadcast(owner, &mut sizes);
    let total = sizes
        .iter()
        .map(|size| usize::try_from(*size).unwrap())
        .sum();
    let mut bytes = if context.rank() == owner {
        encoded.into_iter().flatten().collect()
    } else {
        vec![0; total]
    };
    context.inner.broadcast(owner, &mut bytes);
    if context.rank() != owner {
        let mut offset = 0;
        *blocks = sizes
            .into_iter()
            .map(|size| {
                let size = usize::try_from(size).unwrap();
                let block = decode(&bytes[offset..offset + size]);
                offset += size;
                block
            })
            .collect();
    }
}

fn broadcast_dense<T: Wire + Clone>(
    context: &Context<'_>,
    owner: usize,
    blocks: &mut [Vec<T>],
) {
    let mut bytes = Vec::with_capacity(blocks.iter().map(Vec::len).sum::<usize>() * T::WIDTH);
    if context.rank() == owner {
        for value in blocks.iter().flatten() {
            value.encode(&mut bytes);
        }
    } else {
        bytes.resize(blocks.iter().map(Vec::len).sum::<usize>() * T::WIDTH, 0);
    }
    context.inner.broadcast(owner, &mut bytes);
    if context.rank() != owner {
        let mut values = bytes.chunks_exact(T::WIDTH);
        for value in blocks.iter_mut().flatten() {
            *value = T::decode(values.next().unwrap());
        }
    }
}

fn schedule(edge: usize, layers: Layers) -> (usize, usize, Layers) {
    assert!(edge > 0 && layers.count > 0 && layers.index < layers.count);
    if edge >= layers.count && edge % layers.count == 0 {
        (layers.count, layers.index, Layers { count: 1, index: 0 })
    } else if edge < layers.count && layers.count % edge == 0 {
        (
            edge,
            layers.index % edge,
            Layers {
                count: layers.count / edge,
                index: layers.index / edge,
            },
        )
    } else {
        (1, 0, layers)
    }
}

fn csr_empty<T: Clone>(block: &Csr<T>) -> Csr<T> {
    let (rows, columns) = block.shape();
    Coo::new(rows, columns, Vec::new()).to_csr()
}

fn ccsr_empty<T: Clone>(block: &Ccsr<T>) -> Ccsr<T> {
    let (rows, columns) = block.shape();
    Coo::new(rows, columns, Vec::new()).to_ccsr()
}

fn csr_operand<T: Wire + Clone>(
    plan: Panel<'_, '_>,
    data: &[Csr<T>],
    step: usize,
    edge: usize,
) -> Vec<Csr<T>> {
    let positions = plan.operand_positions(data.len(), step, edge);
    let mut blocks: Vec<_> = positions.iter().map(|&position| data[position].to_coo()).collect();
    if let Some(context) = plan.comm {
        broadcast_sparse(context, step % context.size(), &mut blocks);
    }
    blocks.into_iter().map(|block| block.to_csr()).collect()
}

fn ccsr_operand<T: Wire + Clone>(
    plan: Panel<'_, '_>,
    data: &[Ccsr<T>],
    step: usize,
    edge: usize,
) -> Vec<Ccsr<T>> {
    let positions = plan.operand_positions(data.len(), step, edge);
    let mut blocks: Vec<_> = positions.iter().map(|&position| data[position].to_coo()).collect();
    if let Some(context) = plan.comm {
        broadcast_sparse(context, step % context.size(), &mut blocks);
    }
    blocks.into_iter().map(|block| block.to_ccsr()).collect()
}

fn dense_operand<T: Wire + Clone>(
    plan: Panel<'_, '_>,
    data: &[Vec<T>],
    step: usize,
    edge: usize,
) -> Vec<Vec<T>> {
    let positions = plan.operand_positions(data.len(), step, edge);
    let mut blocks: Vec<_> = positions.iter().map(|&position| data[position].clone()).collect();
    if let Some(context) = plan.comm {
        broadcast_dense(context, step % context.size(), &mut blocks);
    }
    blocks
}

/// Execute a CSR/CSR -> CSR 2D level. The callback is the next contraction
/// level (or the native sparse leaf) and returns its possibly resized sparse
/// output blocks. Sparse moving output follows source ownership: each panel is
/// reduced to its cyclic owner and the old C blocks are structurally added
/// unscaled after reassembly; top-level sparse beta scaling lives outside this
/// source layer.
pub fn execute_csr<A: Semiring>(
    algebra: &A,
    edge: usize,
    layers: Layers,
    a_plan: Panel<'_, '_>,
    b_plan: Panel<'_, '_>,
    c_plan: Panel<'_, '_>,
    a: &[Csr<A::Element>],
    b: &[Csr<A::Element>],
    mut c: Vec<Csr<A::Element>>,
    beta: A::Element,
    mut child: impl FnMut(
        &[Csr<A::Element>],
        &[Csr<A::Element>],
        Vec<Csr<A::Element>>,
        A::Element,
        Layers,
    ) -> Vec<Csr<A::Element>>,
) -> Vec<Csr<A::Element>>
where
    A::Element: Wire,
{
    assert!(!(a_plan.comm.is_some() && b_plan.comm.is_some() && c_plan.comm.is_some()));
    let (count, index, next) = schedule(edge, layers);
    let mut child_beta = beta.clone();
    let moving_output = c_plan.comm.is_some();
    let mut contributions: Vec<_> = if moving_output { c.iter().map(csr_empty).collect() } else { Vec::new() };

    for step in (index..edge).step_by(count) {
        let op_a = csr_operand(a_plan, a, step, edge);
        let op_b = csr_operand(b_plan, b, step, edge);
        if let Some(context) = c_plan.comm {
            assert_eq!(edge % context.size(), 0);
            let positions = c_plan.operand_positions(c.len(), step, edge);
            let work: Vec<_> = positions.iter().map(|&position| csr_empty(&c[position])).collect();
            let work = child(&op_a, &op_b, work, algebra.zero(), next);
            assert_eq!(work.len(), positions.len());
            let owner = step % context.size();
            for (position, block) in positions.into_iter().zip(work) {
                let reduced = block.reduce(context, owner, algebra);
                if context.rank() == owner {
                    contributions[position] = reduced.unwrap();
                }
            }
        } else if c_plan.inner == 0 {
            c = child(&op_a, &op_b, c, child_beta, next);
            child_beta = algebra.one();
        } else {
            let positions = c_plan.operand_positions(c.len(), step, edge);
            if c_plan.outer == 1 {
                let work: Vec<_> = positions.iter().map(|&position| c[position].clone()).collect();
                let work = child(&op_a, &op_b, work, beta.clone(), next);
                assert_eq!(work.len(), positions.len());
                for (position, block) in positions.into_iter().zip(work) {
                    c[position] = block;
                }
            } else {
                let work: Vec<_> = positions.iter().map(|&position| csr_empty(&c[position])).collect();
                let work = child(&op_a, &op_b, work, algebra.zero(), next);
                assert_eq!(work.len(), positions.len());
                for (position, block) in positions.into_iter().zip(work) {
                    c[position] = block;
                }
            }
        }
    }

    if moving_output {
        for (old, contribution) in c.iter_mut().zip(contributions) {
            *old = old.add(&contribution, algebra);
        }
    }
    c
}

/// Execute the native CCSR/dense -> CCSR 2D level. Dense input blocks retain
/// their existing fixed shapes during broadcasts; sparse A and C keep variable
/// encoded sizes. Moving sparse C uses the same source beta convention as
/// [`execute_csr`].
pub fn execute_ccsr_dense<A: Semiring>(
    algebra: &A,
    edge: usize,
    layers: Layers,
    a_plan: Panel<'_, '_>,
    b_plan: Panel<'_, '_>,
    c_plan: Panel<'_, '_>,
    a: &[Ccsr<A::Element>],
    b: &[Vec<A::Element>],
    mut c: Vec<Ccsr<A::Element>>,
    beta: A::Element,
    mut child: impl FnMut(
        &[Ccsr<A::Element>],
        &[Vec<A::Element>],
        Vec<Ccsr<A::Element>>,
        A::Element,
        Layers,
    ) -> Vec<Ccsr<A::Element>>,
) -> Vec<Ccsr<A::Element>>
where
    A::Element: Wire,
{
    assert!(!(a_plan.comm.is_some() && b_plan.comm.is_some() && c_plan.comm.is_some()));
    let (count, index, next) = schedule(edge, layers);
    let mut child_beta = beta.clone();
    let moving_output = c_plan.comm.is_some();
    let mut contributions: Vec<_> = if moving_output { c.iter().map(ccsr_empty).collect() } else { Vec::new() };

    for step in (index..edge).step_by(count) {
        let op_a = ccsr_operand(a_plan, a, step, edge);
        let op_b = dense_operand(b_plan, b, step, edge);
        if let Some(context) = c_plan.comm {
            assert_eq!(edge % context.size(), 0);
            let positions = c_plan.operand_positions(c.len(), step, edge);
            let work: Vec<_> = positions.iter().map(|&position| ccsr_empty(&c[position])).collect();
            let work = child(&op_a, &op_b, work, algebra.zero(), next);
            assert_eq!(work.len(), positions.len());
            let owner = step % context.size();
            for (position, block) in positions.into_iter().zip(work) {
                let reduced = block.reduce(context, owner, algebra);
                if context.rank() == owner {
                    contributions[position] = reduced.unwrap();
                }
            }
        } else if c_plan.inner == 0 {
            c = child(&op_a, &op_b, c, child_beta, next);
            child_beta = algebra.one();
        } else {
            let positions = c_plan.operand_positions(c.len(), step, edge);
            if c_plan.outer == 1 {
                let work: Vec<_> = positions.iter().map(|&position| c[position].clone()).collect();
                let work = child(&op_a, &op_b, work, beta.clone(), next);
                assert_eq!(work.len(), positions.len());
                for (position, block) in positions.into_iter().zip(work) {
                    c[position] = block;
                }
            } else {
                let work: Vec<_> = positions.iter().map(|&position| ccsr_empty(&c[position])).collect();
                let work = child(&op_a, &op_b, work, algebra.zero(), next);
                assert_eq!(work.len(), positions.len());
                for (position, block) in positions.into_iter().zip(work) {
                    c[position] = block;
                }
            }
        }
    }

    if moving_output {
        for (old, contribution) in c.iter_mut().zip(contributions) {
            *old = old.add(&contribution, algebra);
        }
    }
    c
}
