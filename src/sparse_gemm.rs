// Adapted from cc4s CTF spctr_2d_general and sparse CSR contraction kernels.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Semiring, Wire}, context::Context,
    mapping::{Distribution, Mapping, Topology}, sparse::SparseTensor,
    sparse_formats::{Coo, Csr}, tensor::Tensor};

#[path = "sparse_gemm_kernel.rs"]
mod kernel;

struct Layout {
    shape: [usize; 3],
    grid: [usize; 2],
    phase: usize,
    local: [usize; 3],
}
impl Layout {
    fn new(a: &Distribution, b: &Distribution, c: &Distribution, grid: [usize; 2], np: usize) -> Self {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(a.shape[1], b.shape[0]);
        assert_eq!(c.shape, vec![a.shape[0], b.shape[1]]);
        assert!(grid[0] > 0 && grid[1] > 0);
        assert_eq!(grid[0] * grid[1], np);
        let (mut x, mut y) = (grid[0], grid[1]);
        while y != 0 { (x, y) = (y, x % y); }
        let phase = grid[0] / x * grid[1];
        let shape = [a.shape[0], a.shape[1], b.shape[1]];
        Self { shape, grid, phase, local: [shape[0].div_ceil(grid[0]),
            shape[1].div_ceil(phase), shape[2].div_ceil(grid[1])] }
    }
    fn distribution(&self, operand: usize) -> Distribution {
        let topology = Topology::new(self.grid.to_vec());
        let mut maps = vec![Mapping::Unmapped; 2];
        maps[0].augment_physical(&topology, 0);
        maps[1].augment_physical(&topology, 1);
        let shape = match operand {
            0 => { maps[1].augment_virtual(self.phase); vec![self.shape[0], self.shape[1]] }
            1 => { maps[0].augment_virtual(self.phase); vec![self.shape[1], self.shape[2]] }
            2 => vec![self.shape[0], self.shape[2]],
            _ => unreachable!(),
        };
        Distribution::new(shape, topology, maps)
    }
    fn output_pairs<E: Clone>(&self, matrix: &Csr<E>, rank: usize) -> Vec<(usize, E)> {
        let coo = matrix.to_coo();
        let mut pairs: Vec<_> = coo.entries().iter().filter_map(|(row, col, value)| {
            let i = (row - 1) * self.grid[0] + rank % self.grid[0];
            let j = (col - 1) * self.grid[1] + rank / self.grid[0];
            (i < self.shape[0] && j < self.shape[2]).then(|| (i + self.shape[0] * j, value.clone()))
        }).collect();
        pairs.sort_by_key(|pair| pair.0);
        pairs
    }
}

// Sparse panels carry their variable entry count before the serialized pairs.
fn broadcast<E: Wire + Clone>(context: &Context<'_>, root: usize,
    entries: Vec<(usize, usize, E)>, rows: usize, cols: usize) -> Csr<E> {
    let mut count = [entries.len() as u64];
    context.broadcast(root, &mut count);
    let width = 16 + E::WIDTH;
    let mut bytes = Vec::with_capacity(count[0] as usize * width);
    if context.rank() == root {
        for (row, col, value) in entries {
            (row as u64).encode(&mut bytes);
            (col as u64).encode(&mut bytes);
            value.encode(&mut bytes);
        }
    } else { bytes.resize(count[0] as usize * width, 0); }
    context.inner.broadcast(root, &mut bytes);
    let entries = bytes.chunks_exact(width).map(|entry| (
        u64::decode(&entry[..8]) as usize, u64::decode(&entry[8..16]) as usize,
        E::decode(&entry[16..]))).collect();
    Coo::new(rows, cols, entries).to_csr()
}

fn sparse_panels<A: Semiring + Clone>(a: &SparseTensor<'_, '_, A>, b: &SparseTensor<'_, '_, A>,
    layout: &Layout, mut child: impl FnMut(&Csr<A::Element>, &Csr<A::Element>, usize))
where A::Element: Wire {
    let mut a = a.clone();
    let mut b = b.clone();
    a.redistribute(layout.distribution(0));
    b.redistribute(layout.distribution(1));
    let context = a.context();
    let rank = context.rank();
    let row = context.split(Some((rank % layout.grid[0]) as i32), rank as i32).unwrap();
    let col = context.split(Some((rank / layout.grid[0]) as i32), rank as i32).unwrap();
    let aa = a.local_pairs();
    let bb = b.local_pairs();
    for step in 0..layout.phase {
        let ae = aa.iter().filter_map(|(key, value)| {
            let i = key % layout.shape[0];
            let k = key / layout.shape[0];
            (k % layout.phase == step).then(|| (i / layout.grid[0] + 1,
                k / layout.phase + 1, value.clone()))
        }).collect();
        let be = bb.iter().filter_map(|(key, value)| {
            let k = key % layout.shape[1];
            let j = key / layout.shape[1];
            (k % layout.phase == step).then(|| (k / layout.phase + 1,
                j / layout.grid[1] + 1, value.clone()))
        }).collect();
        let a_panel = broadcast(&row, step % layout.grid[1], ae, layout.local[0], layout.local[1]);
        let b_panel = broadcast(&col, step % layout.grid[0], be, layout.local[1], layout.local[2]);
        child(&a_panel, &b_panel, step);
    }
    row.close();
    col.close();
}

impl<A: Semiring + Clone> SparseTensor<'_, '_, A> where A::Element: Wire {
    /// Explicit-grid sparse matrix product, with sparse output throughout.
    pub fn gemm_sparse(&mut self, a: &Self, b: &Self, grid: [usize; 2],
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let entries = self.local_pairs().into_iter().map(|(key, value)| (
            key % layout.shape[0] / grid[0] + 1,
            key / layout.shape[0] / grid[1] + 1, value)).collect();
        let mut c = Coo::new(layout.local[0], layout.local[2], entries).to_csr();
        let algebra = self.algebra().clone();
        let one = algebra.one();
        sparse_panels(a, b, &layout, |a, b, step| {
            c = a.multiply_sparse(b, &alpha, if step == 0 { &beta } else { &one }, Some(&c), &algebra);
        });
        self.blocks = vec![layout.output_pairs(&c, self.context().rank())];
        self.redistribute(original);
    }
}

impl<A: Semiring + Clone> Tensor<'_, '_, A> where A::Element: Wire {
    /// Sparse-by-sparse matrix product into distributed dense storage.
    pub fn gemm_sparse(&mut self, a: &SparseTensor<'_, '_, A>, b: &SparseTensor<'_, '_, A>,
        grid: [usize; 2], alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let mut values = self.local_storage().to_vec();
        let algebra = self.algebra().clone();
        let one = algebra.one();
        sparse_panels(a, b, &layout, |a, b, step| {
            kernel::csr_sparse_dense(a, b, &mut values, &alpha,
                if step == 0 { &beta } else { &one }, &algebra);
        });
        let dist = self.distribution().clone();
        let rank = self.context().rank();
        self.transform(|key, value| *value = values[dist.local_offset(rank, key)].clone());
        self.redistribute(original);
    }

    /// Sparse-by-dense matrix product with variable sparse and fixed dense panels.
    pub fn gemm_sparse_dense(&mut self, a: &SparseTensor<'_, '_, A>, b: &Self,
        grid: [usize; 2], alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let mut a = a.clone();
        let mut b = b.clone();
        a.redistribute(layout.distribution(0));
        b.redistribute(layout.distribution(1));
        let rank = self.context().rank();
        let context = self.context();
        let row = context.split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = context.split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let aa = a.local_pairs();
        let bb = b.local_pairs();
        let algebra = self.algebra().clone();
        let one = algebra.one();
        let mut values = self.local_storage().to_vec();
        for step in 0..layout.phase {
            let ae = aa.iter().filter_map(|(key, value)| {
                let i = key % layout.shape[0];
                let k = key / layout.shape[0];
                (k % layout.phase == step).then(|| (i / grid[0] + 1, k / layout.phase + 1, value.clone()))
            }).collect();
            let a_panel = broadcast(&row, step % grid[1], ae, layout.local[0], layout.local[1]);
            let mut b_panel = vec![algebra.zero(); layout.local[1] * layout.local[2]];
            if col.rank() == step % grid[0] {
                for (key, value) in &bb {
                    let k = key % layout.shape[1];
                    let j = key / layout.shape[1];
                    if k % layout.phase == step {
                        b_panel[k / layout.phase + (j / grid[1]) * layout.local[1]] = value.clone();
                    }
                }
            }
            col.broadcast(step % grid[0], &mut b_panel);
            a_panel.multiply_dense(layout.local[2], &b_panel, &alpha,
                if step == 0 { &beta } else { &one }, &mut values, &algebra);
        }
        row.close();
        col.close();
        let dist = self.distribution().clone();
        self.transform(|key, value| *value = values[dist.local_offset(rank, key)].clone());
        self.redistribute(original);
    }

    /// Dense-by-sparse matrix product with fixed dense and variable sparse panels.
    pub fn gemm_dense_sparse(&mut self, a: &Self, b: &SparseTensor<'_, '_, A>,
        grid: [usize; 2], alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let mut a = a.clone();
        let mut b = b.clone();
        a.redistribute(layout.distribution(0));
        b.redistribute(layout.distribution(1));
        let rank = self.context().rank();
        let context = self.context();
        let row = context.split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = context.split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let aa = a.local_pairs();
        let bb = b.local_pairs();
        let algebra = self.algebra().clone();
        let one = algebra.one();
        let mut values = self.local_storage().to_vec();
        for step in 0..layout.phase {
            let mut a_panel = vec![algebra.zero(); layout.local[0] * layout.local[1]];
            if row.rank() == step % grid[1] {
                for (key, value) in &aa {
                    let i = key % layout.shape[0];
                    let k = key / layout.shape[0];
                    if k % layout.phase == step {
                        a_panel[i / grid[0] + (k / layout.phase) * layout.local[0]] = value.clone();
                    }
                }
            }
            row.broadcast(step % grid[1], &mut a_panel);
            let be = bb.iter().filter_map(|(key, value)| {
                let k = key % layout.shape[1];
                let j = key / layout.shape[1];
                (k % layout.phase == step).then(|| (k / layout.phase + 1,
                    j / grid[1] + 1, value.clone()))
            }).collect();
            let b_panel = broadcast(&col, step % grid[0], be, layout.local[1], layout.local[2]);
            if step == 0 && beta != one {
                for value in &mut values {
                    *value = algebra.multiply(&beta, value);
                }
            }
            for k in 0..layout.local[1] {
                let row_start = b_panel.row_offsets()[k] - 1;
                let row_end = b_panel.row_offsets()[k + 1] - 1;
                for entry in row_start..row_end {
                    let j = b_panel.columns()[entry] - 1;
                    for i in 0..layout.local[0] {
                        let product = algebra.multiply(
                            &a_panel[i + k * layout.local[0]],
                            &b_panel.values()[entry],
                        );
                        let product = if alpha != one {
                            algebra.multiply(&alpha, &product)
                        } else {
                            product
                        };
                        let index = i + j * layout.local[0];
                        values[index] = algebra.add(&values[index], &product);
                    }
                }
            }
        }
        row.close();
        col.close();
        let dist = self.distribution().clone();
        self.transform(|key, value| *value = values[dist.local_offset(rank, key)].clone());
        self.redistribute(original);
    }
}
