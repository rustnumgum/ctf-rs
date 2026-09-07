// Adapted from cc4s CTF spctr_2d_general and sparse CSR contraction kernels.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Semiring, Wire}, context::Context,
    mapping::{Distribution, Mapping, Topology}, sparse::SparseTensor,
    sparse_2d::{self, Layers, Panel},
    sparse_formats::{Coo, Csr},
    sparse_matricize::{dematricize_pairs, matricize_pairs, Dematricization, Matricization},
    symmetry::Symmetry, tensor::Tensor};

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
    fn matricization(&self, operand: usize) -> Matricization {
        let (shape, phases, folded_shape) = match operand {
            0 => (
                vec![self.shape[0], self.shape[1]],
                vec![self.grid[0], self.phase],
                vec![self.local[0], self.local[1]],
            ),
            1 => (
                vec![self.shape[1], self.shape[2]],
                vec![self.phase, self.grid[1]],
                vec![self.local[1], self.local[2]],
            ),
            2 => (
                vec![self.shape[0], self.shape[2]],
                vec![self.grid[0], self.grid[1]],
                vec![self.local[0], self.local[2]],
            ),
            _ => unreachable!(),
        };
        let padded_shape = shape
            .iter()
            .zip(&phases)
            .map(|(&length, &phase)| length.div_ceil(phase) * phase)
            .collect();
        Matricization {
            shape,
            padded_shape,
            links: vec![Symmetry::NS; 2],
            folded_shape,
            reverse_ordering: vec![0, 1],
            row_dimensions: 1,
            phases,
        }
    }
    fn output_pairs<E: Clone>(&self, matrix: &Coo<E>, rank: usize) -> Vec<(usize, E)> {
        let physical = [rank % self.grid[0], rank / self.grid[0]];
        let valid_rows = if physical[0] < self.shape[0] {
            (self.shape[0] - 1 - physical[0]) / self.grid[0] + 1
        } else {
            0
        };
        let valid_columns = if physical[1] < self.shape[2] {
            (self.shape[2] - 1 - physical[1]) / self.grid[1] + 1
        } else {
            0
        };
        let filtered = Coo::new(
            matrix.shape().0,
            matrix.shape().1,
            matrix
                .entries()
                .iter()
                .filter(|(row, column, _)| *row <= valid_rows && *column <= valid_columns)
                .cloned()
                .collect(),
        );
        dematricize_pairs(
            &Dematricization {
                shape: vec![self.shape[0], self.shape[2]],
                reverse_ordering: vec![0, 1],
                row_dimensions: 1,
                phases: vec![self.grid[0], self.grid[1]],
                phase_ranks: physical.to_vec(),
            },
            &filtered,
        )
    }
}

fn dense_blocks<E: Clone>(values: &[E], count: usize, block_size: usize) -> Vec<Vec<E>> {
    assert_eq!(values.len(), count * block_size);
    if block_size == 0 {
        vec![Vec::new(); count]
    } else {
        values.chunks_exact(block_size).map(<[E]>::to_vec).collect()
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

impl<A: Semiring + Clone> SparseTensor<'_, '_, A> where A::Element: Wire {
    /// Explicit-grid sparse matrix product, with sparse output throughout.
    pub fn gemm_sparse(&mut self, a: &Self, b: &Self, grid: [usize; 2],
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let algebra = self.algebra().clone();
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.redistribute(layout.distribution(0));
        bb.redistribute(layout.distribution(1));
        let a_metadata = layout.matricization(0);
        let b_metadata = layout.matricization(1);
        let a_panels: Vec<_> = aa.blocks.iter()
            .map(|block| matricize_pairs(&a_metadata, block).to_csr()).collect();
        let b_panels: Vec<_> = bb.blocks.iter()
            .map(|block| matricize_pairs(&b_metadata, block).to_csr()).collect();
        let c = vec![matricize_pairs(&layout.matricization(2), &self.blocks[0]).to_csr()];
        let rank = self.context().rank();
        let row = self.context().split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = self.context().split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let c = sparse_2d::execute_csr(
            &algebra, layout.phase, Layers { count: 1, index: 0 },
            Panel { comm: Some(&row), outer: 1, inner: 1 },
            Panel { comm: Some(&col), outer: 1, inner: 1 },
            Panel { comm: None, outer: 1, inner: 0 },
            &a_panels, &b_panels, c, beta,
            |a, b, mut c, leaf_beta, _| {
                c[0] = a[0].multiply_sparse(
                    &b[0], &alpha, &leaf_beta, Some(&c[0]), &algebra,
                );
                c
            },
        );
        row.close();
        col.close();
        self.blocks = vec![layout.output_pairs(&c[0].to_coo(), self.context().rank())];
        self.redistribute(original);
    }

    /// Explicit-grid sparse-by-dense matrix product with sparse CCSR output.
    /// Represented rows follow sparse A; structural zero results are retained.
    pub fn gemm_sparse_dense(&mut self, a: &Self, b: &Tensor<'_, '_, A>, grid: [usize; 2],
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let algebra = self.algebra().clone();
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.redistribute(layout.distribution(0));
        bb.redistribute(layout.distribution(1));
        let a_metadata = layout.matricization(0);
        let a_panels: Vec<_> = aa.blocks.iter()
            .map(|block| matricize_pairs(&a_metadata, block).to_ccsr()).collect();
        let b_panels = dense_blocks(
            bb.local_storage(), layout.phase / grid[0], layout.local[1] * layout.local[2],
        );
        let c = vec![matricize_pairs(&layout.matricization(2), &self.blocks[0]).to_ccsr()];
        let rank = self.context().rank();
        let row = self.context().split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = self.context().split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let c = sparse_2d::execute_ccsr_dense(
            &algebra, layout.phase, Layers { count: 1, index: 0 },
            Panel { comm: Some(&row), outer: 1, inner: 1 },
            Panel { comm: Some(&col), outer: 1, inner: 1 },
            Panel { comm: None, outer: 1, inner: 0 },
            &a_panels, &b_panels, c, beta,
            |a, b, mut c, leaf_beta, _| {
                c[0] = a[0].multiply_dense(
                    layout.local[2], &b[0], &alpha, &leaf_beta, Some(&c[0]), &algebra,
                );
                c
            },
        );
        row.close();
        col.close();
        self.blocks = vec![layout.output_pairs(&c[0].to_coo(), self.context().rank())];
        self.redistribute(original);
    }

    /// Dense matrix product through the source dense-output-then-sparsify path.
    /// The source pointer predicate retains every valid dense entry, including
    /// values equal to the additive identity.
    pub fn gemm_dense(&mut self, a: &Tensor<'_, '_, A>, b: &Tensor<'_, '_, A>, grid: [usize; 2],
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let mut dense = self.clone().into_dense();
        dense.contract_from("ij", a, "ik", b, "kj", Topology::new(grid.to_vec()), alpha, beta)
            .unwrap();
        self.blocks = dense.into_sparse(|_| true).blocks;
    }

    /// Ordinary dense-by-sparse product using the source's unconditional
    /// operand swap. For noncommutative multiplication this evaluates B*A.
    pub fn gemm_dense_sparse(&mut self, a: &Tensor<'_, '_, A>, b: &Self, grid: [usize; 2],
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let original = self.distribution().clone();
        let mut c = self.permute_axes(&[1, 0]);
        let b = b.permute_axes(&[1, 0]);
        let a = a.permute_axes(&[1, 0]);
        c.gemm_sparse_dense(&b, &a, [grid[1], grid[0]], alpha, beta);
        let mut restored = c.permute_axes(&[1, 0]);
        restored.redistribute(original);
        *self = restored;
    }

    /// Custom sparse matrix product with sparse output throughout. Structural
    /// products, including zero-valued results, are retained as in source CSR.
    pub fn gemm_sparse_function(&mut self, a: &Self, b: &Self, grid: [usize; 2],
        alpha: A::Element, beta: A::Element,
        function: impl Fn(&A::Element, &A::Element) -> A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let algebra = self.algebra().clone();
        let one = algebra.one();
        assert!(alpha == one,"source custom CSR kernel requires identity alpha");
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.redistribute(layout.distribution(0));
        bb.redistribute(layout.distribution(1));
        let a_metadata = layout.matricization(0);
        let b_metadata = layout.matricization(1);
        let a_panels: Vec<_> = aa.blocks.iter()
            .map(|block| matricize_pairs(&a_metadata, block).to_csr()).collect();
        let b_panels: Vec<_> = bb.blocks.iter()
            .map(|block| matricize_pairs(&b_metadata, block).to_csr()).collect();
        let rank = self.context().rank();
        let row = self.context().split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = self.context().split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let c = sparse_2d::execute_csr(
            &algebra, layout.phase, Layers { count: 1, index: 0 },
            Panel { comm: Some(&row), outer: 1, inner: 1 },
            Panel { comm: Some(&col), outer: 1, inner: 1 },
            Panel { comm: None, outer: 1, inner: 0 },
            &a_panels, &b_panels,
            vec![Coo::new(layout.local[0], layout.local[2], Vec::new()).to_csr()],
            one.clone(),
            |a, b, mut c, _, _| {
                c[0] = crate::sparse_function_kernel::csr_sparse_output(
                    &algebra, &a[0], &b[0], &c[0], &function,
                );
                c
            },
        );
        row.close();
        col.close();
        // Source home_contract computes into empty C_buf, then sparse-sums it
        // into old C. This retains old-only zero keys and right-scales by beta.
        let mut product = Self::new(self.context(),layout.distribution(2),algebra);
        product.blocks = vec![layout.output_pairs(&c[0].to_coo(),self.context().rank())];
        self.sum_from("ij",&product,"ij",one,beta);
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
        let algebra = self.algebra().clone();
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.redistribute(layout.distribution(0));
        bb.redistribute(layout.distribution(1));
        let a_metadata = layout.matricization(0);
        let b_metadata = layout.matricization(1);
        let a_panels: Vec<_> = aa.blocks.iter()
            .map(|block| matricize_pairs(&a_metadata, block).to_csr()).collect();
        let b_panels: Vec<_> = bb.blocks.iter()
            .map(|block| matricize_pairs(&b_metadata, block).to_csr()).collect();
        let rank = self.context().rank();
        let row = self.context().split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = self.context().split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let values = sparse_2d::execute_csr_sparse_dense(
            &algebra, layout.phase, Layers { count: 1, index: 0 },
            Panel { comm: Some(&row), outer: 1, inner: 1 },
            Panel { comm: Some(&col), outer: 1, inner: 1 },
            Panel { comm: None, outer: 1, inner: 0 },
            &a_panels, &b_panels, vec![self.local_storage().to_vec()], beta,
            |a, b, mut c, leaf_beta, _| {
                kernel::csr_sparse_dense(
                    &a[0], &b[0], &mut c[0], &alpha, &leaf_beta, &algebra,
                );
                c
            },
        ).remove(0);
        row.close();
        col.close();
        let dist = self.distribution().clone();
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
        let algebra = self.algebra().clone();
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.redistribute(layout.distribution(0));
        bb.redistribute(layout.distribution(1));
        let a_metadata = layout.matricization(0);
        let a_panels: Vec<_> = aa.blocks.iter()
            .map(|block| matricize_pairs(&a_metadata, block).to_csr()).collect();
        let b_panels = dense_blocks(
            bb.local_storage(), layout.phase / grid[0], layout.local[1] * layout.local[2],
        );
        let rank = self.context().rank();
        let row = self.context().split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = self.context().split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let values = sparse_2d::execute_csr_dense(
            &algebra, layout.phase, Layers { count: 1, index: 0 },
            Panel { comm: Some(&row), outer: 1, inner: 1 },
            Panel { comm: Some(&col), outer: 1, inner: 1 },
            Panel { comm: None, outer: 1, inner: 0 },
            &a_panels, &b_panels, vec![self.local_storage().to_vec()], beta,
            |a, b, mut c, leaf_beta, _| {
                a[0].multiply_dense(
                    layout.local[2], &b[0], &alpha, &leaf_beta, &mut c[0], &algebra,
                );
                c
            },
        ).remove(0);
        row.close();
        col.close();
        let dist = self.distribution().clone();
        self.transform(|key, value| *value = values[dist.local_offset(rank, key)].clone());
        self.redistribute(original);
    }

    /// Sparse-by-sparse matrix product into dense storage using a custom
    /// bivariate function. Missing sparse entries are never evaluated.
    pub fn gemm_sparse_function(
        &mut self,
        a: &SparseTensor<'_, '_, A>,
        b: &SparseTensor<'_, '_, A>,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
        function: impl Fn(&A::Element, &A::Element) -> A::Element,
    ) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let algebra = self.algebra().clone();
        let one = algebra.one();
        assert!(alpha == one,"source custom CSR kernel requires identity alpha");
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.redistribute(layout.distribution(0));
        bb.redistribute(layout.distribution(1));
        let a_metadata = layout.matricization(0);
        let b_metadata = layout.matricization(1);
        let a_panels: Vec<_> = aa.blocks.iter()
            .map(|block| matricize_pairs(&a_metadata, block).to_csr()).collect();
        let b_panels: Vec<_> = bb.blocks.iter()
            .map(|block| matricize_pairs(&b_metadata, block).to_csr()).collect();
        let rank = self.context().rank();
        let row = self.context().split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = self.context().split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let values = sparse_2d::execute_csr_sparse_dense(
            &algebra, layout.phase, Layers { count: 1, index: 0 },
            Panel { comm: Some(&row), outer: 1, inner: 1 },
            Panel { comm: Some(&col), outer: 1, inner: 1 },
            Panel { comm: None, outer: 1, inner: 0 },
            &a_panels, &b_panels, vec![self.local_storage().to_vec()], beta,
            |a, b, mut c, leaf_beta, _| {
                crate::sparse_function_kernel::csr_sparse(
                    &algebra, &a[0], &b[0], &mut c[0], &alpha, &leaf_beta, &function,
                );
                c
            },
        ).remove(0);
        row.close();
        col.close();
        let dist = self.distribution().clone();
        self.transform(|key, value| *value = values[dist.local_offset(rank, key)].clone());
        self.redistribute(original);
    }

    /// Sparse-by-dense matrix product into dense storage using a custom
    /// bivariate function. Stored sparse zeros and all dense values, including
    /// zeros, are evaluated; absent sparse entries are not.
    pub fn gemm_sparse_dense_function(
        &mut self,
        a: &SparseTensor<'_, '_, A>,
        b: &Self,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
        function: impl Fn(&A::Element, &A::Element) -> A::Element,
    ) {
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        let layout = Layout::new(a.distribution(), b.distribution(), self.distribution(), grid, self.context().size());
        let original = self.distribution().clone();
        self.redistribute(layout.distribution(2));
        let algebra = self.algebra().clone();
        let one = algebra.one();
        assert!(alpha == one,"source custom CSR kernel requires identity alpha");
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.redistribute(layout.distribution(0));
        bb.redistribute(layout.distribution(1));
        let a_metadata = layout.matricization(0);
        let a_panels: Vec<_> = aa.blocks.iter()
            .map(|block| matricize_pairs(&a_metadata, block).to_csr()).collect();
        let b_panels = dense_blocks(
            bb.local_storage(), layout.phase / grid[0], layout.local[1] * layout.local[2],
        );
        let rank = self.context().rank();
        let row = self.context().split(Some((rank % grid[0]) as i32), rank as i32).unwrap();
        let col = self.context().split(Some((rank / grid[0]) as i32), rank as i32).unwrap();
        let values = sparse_2d::execute_csr_dense(
            &algebra, layout.phase, Layers { count: 1, index: 0 },
            Panel { comm: Some(&row), outer: 1, inner: 1 },
            Panel { comm: Some(&col), outer: 1, inner: 1 },
            Panel { comm: None, outer: 1, inner: 0 },
            &a_panels, &b_panels, vec![self.local_storage().to_vec()], beta,
            |a, b, mut c, leaf_beta, _| {
                crate::sparse_function_kernel::csr_dense(
                    &algebra, &a[0], layout.local[2], &b[0], &mut c[0],
                    &alpha, &leaf_beta, &function,
                );
                c
            },
        ).remove(0);
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
