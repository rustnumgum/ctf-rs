// Adapted from cc4s CTF contraction/contraction.cxx folding, get_len_ordering,
// calc_fold_lnmk, and sparse spmatricize paths.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Fully folded nonsymmetric sparse tensor contractions.
//!
//! Unique labels are classified as AB=`k`, AC=`m`, BC=`n`, and ABC=`l`.
//! Axes are matricized as `A[m,k,l]`, `B[k,n,l]`, and `C[m,n,l]`; every
//! `l` coordinate is then executed by the existing distributed sparse GEMM.
//! Repeated labels are projected onto diagonals before folding; output
//! reinsertion preserves off-diagonal values. One-operand-only labels are
//! not supported by this folded path.

use crate::{
    algebra::{Semiring, Wire},
    diagonal::Projection,
    folding::{Operand, Rejected},
    mapping::Distribution,
    sparse::SparseTensor,
    tensor::Tensor,
};

struct Plan {
    order_a: Vec<usize>,
    order_b: Vec<usize>,
    order_c: Vec<usize>,
    inverse_c: Vec<usize>,
    canonical_c_shape: Vec<usize>,
    m: usize,
    n: usize,
    k: usize,
    batches: usize,
}

impl Plan {
    fn new(shapes: [&[usize]; 3], indices: [&str; 3]) -> Result<Self, Rejected> {
        let operands = [Operand::A, Operand::B, Operand::C];
        for operand in 0..3 {
            if !indices[operand].is_ascii() {
                return Err(Rejected::NonAsciiIndices {
                    operand: operands[operand],
                });
            }
            if shapes[operand].len() != indices[operand].len() {
                return Err(Rejected::RankMismatch {
                    operand: operands[operand],
                    shape: shapes[operand].len(),
                    indices: indices[operand].len(),
                });
            }
            let mut seen = [false; 256];
            for label in indices[operand].bytes() {
                if seen[label as usize] {
                    return Err(Rejected::RepeatedIndex {
                        operand: operands[operand],
                        label: label as char,
                    });
                }
                seen[label as usize] = true;
            }
        }

        let mut masks = [0_u8; 256];
        let mut dimensions = [None; 256];
        for operand in 0..3 {
            for (&dimension, label) in shapes[operand].iter().zip(indices[operand].bytes()) {
                if dimensions[label as usize].is_some_and(|old| old != dimension) {
                    return Err(Rejected::DimensionMismatch {
                        label: label as char,
                    });
                }
                dimensions[label as usize] = Some(dimension);
                masks[label as usize] |= 1 << operand;
            }
        }
        for (label, &mask) in masks.iter().enumerate() {
            let operand = match mask {
                0 | 0b011 | 0b101 | 0b110 | 0b111 => continue,
                0b001 => Operand::A,
                0b010 => Operand::B,
                0b100 => Operand::C,
                _ => unreachable!(),
            };
            return Err(Rejected::OneOperandLabel {
                operand,
                label: label as u8 as char,
            });
        }
        for shape in shapes {
            checked_product(shape)?;
        }

        let labels_a = indices[0].as_bytes();
        let labels_b = indices[1].as_bytes();
        let labels_c = indices[2].as_bytes();
        let m_labels = labels_with_mask(labels_a, &masks, 0b101);
        let k_labels = labels_with_mask(labels_a, &masks, 0b011);
        let n_labels = labels_with_mask(labels_b, &masks, 0b110);
        let l_labels = labels_with_mask(labels_a, &masks, 0b111);
        let m = label_product(&m_labels, &dimensions)?;
        let n = label_product(&n_labels, &dimensions)?;
        let k = label_product(&k_labels, &dimensions)?;
        let batches = label_product(&l_labels, &dimensions)?;
        for (left, right) in [(m, k), (k, n), (m, n)] {
            left.checked_mul(right)
                .and_then(|size| size.checked_mul(batches))
                .ok_or(Rejected::SizeOverflow)?;
        }

        let packed_a = concatenate(&[&m_labels, &k_labels, &l_labels]);
        let packed_b = concatenate(&[&k_labels, &n_labels, &l_labels]);
        let packed_c = concatenate(&[&m_labels, &n_labels, &l_labels]);
        let order_a = axis_order(labels_a, &packed_a);
        let order_b = axis_order(labels_b, &packed_b);
        let order_c = axis_order(labels_c, &packed_c);
        let canonical_c_shape = order_c.iter().map(|&axis| shapes[2][axis]).collect();
        let mut inverse_c = vec![0; order_c.len()];
        for (canonical, &original) in order_c.iter().enumerate() {
            inverse_c[original] = canonical;
        }

        Ok(Self {
            order_a,
            order_b,
            order_c,
            inverse_c,
            canonical_c_shape,
            m,
            n,
            k,
            batches,
        })
    }
}

fn labels_with_mask(source: &[u8], masks: &[u8; 256], mask: u8) -> Vec<u8> {
    source
        .iter()
        .copied()
        .filter(|&label| masks[label as usize] == mask)
        .collect()
}

fn label_product(
    labels: &[u8],
    dimensions: &[Option<usize>; 256],
) -> Result<usize, Rejected> {
    labels.iter().try_fold(1usize, |product, &label| {
        product
            .checked_mul(dimensions[label as usize].unwrap())
            .ok_or(Rejected::SizeOverflow)
    })
}

fn checked_product(shape: &[usize]) -> Result<usize, Rejected> {
    shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(Rejected::SizeOverflow)
    })
}

fn concatenate(groups: &[&[u8]]) -> Vec<u8> {
    groups.iter().flat_map(|group| group.iter().copied()).collect()
}

fn axis_order(source: &[u8], packed: &[u8]) -> Vec<usize> {
    packed
        .iter()
        .map(|label| source.iter().position(|candidate| candidate == label).unwrap())
        .collect()
}

fn validate_grid(grid: [usize; 2], processes: usize) {
    assert!(grid[0] > 0 && grid[1] > 0);
    assert_eq!(grid[0].checked_mul(grid[1]), Some(processes));
}

// Preserve typed validation errors before any diagonal redistribution occurs.
fn repeated_projections(shapes: [&[usize]; 3], indices: [&str; 3])
    -> Result<Option<[Projection; 3]>, Rejected> {
    if !indices.iter().any(|labels| labels.bytes().enumerate()
        .any(|(axis, label)| labels.as_bytes()[..axis].contains(&label))) { return Ok(None); }
    let operands = [Operand::A, Operand::B, Operand::C];
    let mut dimensions = [None; 256];
    for operand in 0..3 {
        if !indices[operand].is_ascii() { return Err(Rejected::NonAsciiIndices { operand: operands[operand] }); }
        if shapes[operand].len() != indices[operand].len() {
            return Err(Rejected::RankMismatch { operand: operands[operand],
                shape: shapes[operand].len(), indices: indices[operand].len() });
        }
        for (&dimension, label) in shapes[operand].iter().zip(indices[operand].bytes()) {
            if dimensions[label as usize].is_some_and(|old| old != dimension) {
                return Err(Rejected::DimensionMismatch { label: label as char });
            }
            dimensions[label as usize] = Some(dimension);
        }
    }
    let projections = std::array::from_fn(|operand| Projection::new(shapes[operand], indices[operand]));
    let [a, b, c] = &projections;
    Plan::new([&a.shape, &b.shape, &c.shape], [&a.labels, &b.labels, &c.labels])?;
    Ok(Some(projections))
}

impl<'c, 'r, A: Semiring + Clone> SparseTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Contract fully foldable sparse tensors, including diagonal labels,
    /// without gathering.
    pub fn contract_from(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
    ) -> Result<(), Rejected> {
        if let Some([_, _, output]) = repeated_projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c])? {
            assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
            validate_grid(grid, self.context().size());
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_from(&ic, &aa, &ia, &bb, &ib, grid, alpha, beta)?;
            if output.repeated() { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return Ok(());
        }
        let plan = Plan::new(
            [
                &a.distribution().shape,
                &b.distribution().shape,
                &self.distribution().shape,
            ],
            [indices_a, indices_b, indices_c],
        )?;
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        validate_grid(grid, self.context().size());

        let processes = self.context().size();
        let original = self.distribution().clone();
        let a = a.permute_axes(&plan.order_a).reshape(Distribution::cyclic(
            vec![plan.m, plan.k, plan.batches],
            processes,
        ));
        let b = b.permute_axes(&plan.order_b).reshape(Distribution::cyclic(
            vec![plan.k, plan.n, plan.batches],
            processes,
        ));
        let c = self
            .permute_axes(&plan.order_c)
            .reshape(Distribution::cyclic(
                vec![plan.m, plan.n, plan.batches],
                processes,
            ));
        let mut folded = Self::new(
            self.context(),
            Distribution::cyclic(vec![plan.m, plan.n, plan.batches], processes),
            self.algebra().clone(),
        );
        let matrix_len = plan.m.checked_mul(plan.n).unwrap();
        for batch in 0..plan.batches {
            let aa = a
                .slice(&[0..plan.m, 0..plan.k, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.k], processes));
            let bb = b
                .slice(&[0..plan.k, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.k, plan.n], processes));
            let mut cc = c
                .slice(&[0..plan.m, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.n], processes));
            cc.gemm_sparse(&aa, &bb, grid, alpha.clone(), beta.clone());
            let pairs: Vec<_> = cc
                .local_pairs()
                .into_iter()
                .filter(|(key, _)| cc.distribution().owner(*key) == self.context().rank())
                .map(|(key, value)| (key + batch * matrix_len, value))
                .collect();
            folded.write_add(&pairs);
        }

        let mut restored = folded
            .reshape(Distribution::cyclic(plan.canonical_c_shape, processes))
            .permute_axes(&plan.inverse_c);
        restored.redistribute(original);
        *self = restored;
        Ok(())
    }
}

impl<'c, 'r, A: Semiring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Contract sparse A and B into a dense fully folded output.
    pub fn contract_from_sparse(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &SparseTensor<'_, '_, A>,
        indices_b: &str,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
    ) -> Result<(), Rejected> {
        if let Some([_, _, output]) = repeated_projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c])? {
            assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
            validate_grid(grid, self.context().size());
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_from_sparse(&ic, &aa, &ia, &bb, &ib, grid, alpha, beta)?;
            if output.repeated() { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return Ok(());
        }
        let plan = Plan::new(
            [
                &a.distribution().shape,
                &b.distribution().shape,
                &self.distribution().shape,
            ],
            [indices_a, indices_b, indices_c],
        )?;
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        validate_grid(grid, self.context().size());

        let processes = self.context().size();
        let original = self.distribution().clone();
        let a = a.permute_axes(&plan.order_a).reshape(Distribution::cyclic(
            vec![plan.m, plan.k, plan.batches],
            processes,
        ));
        let b = b.permute_axes(&plan.order_b).reshape(Distribution::cyclic(
            vec![plan.k, plan.n, plan.batches],
            processes,
        ));
        let c = self
            .permute_axes(&plan.order_c)
            .reshape(Distribution::cyclic(
                vec![plan.m, plan.n, plan.batches],
                processes,
            ));
        let mut folded = Self::new(
            self.context(),
            Distribution::cyclic(vec![plan.m, plan.n, plan.batches], processes),
            self.algebra().clone(),
        );
        let matrix_len = plan.m.checked_mul(plan.n).unwrap();
        for batch in 0..plan.batches {
            let aa = a
                .slice(&[0..plan.m, 0..plan.k, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.k], processes));
            let bb = b
                .slice(&[0..plan.k, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.k, plan.n], processes));
            let mut cc = c
                .slice(&[0..plan.m, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.n], processes));
            cc.gemm_sparse(&aa, &bb, grid, alpha.clone(), beta.clone());
            let pairs: Vec<_> = cc
                .local_pairs()
                .into_iter()
                .filter(|(key, _)| cc.distribution().owner(*key) == self.context().rank())
                .map(|(key, value)| (key + batch * matrix_len, value))
                .collect();
            folded.write_add(&pairs);
        }

        let mut restored = folded
            .reshape(Distribution::cyclic(plan.canonical_c_shape, processes))
            .permute_axes(&plan.inverse_c);
        restored.redistribute(original);
        *self = restored;
        Ok(())
    }

    /// Contract sparse A and dense B into a dense fully folded output.
    pub fn contract_from_sparse_dense(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
    ) -> Result<(), Rejected> {
        if let Some([_, _, output]) = repeated_projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c])? {
            assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
            validate_grid(grid, self.context().size());
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_from_sparse_dense(&ic, &aa, &ia, &bb, &ib, grid, alpha, beta)?;
            if output.repeated() { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return Ok(());
        }
        let plan = Plan::new(
            [
                &a.distribution().shape,
                &b.distribution().shape,
                &self.distribution().shape,
            ],
            [indices_a, indices_b, indices_c],
        )?;
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        validate_grid(grid, self.context().size());

        let processes = self.context().size();
        let original = self.distribution().clone();
        let a = a.permute_axes(&plan.order_a).reshape(Distribution::cyclic(
            vec![plan.m, plan.k, plan.batches],
            processes,
        ));
        let b = b.permute_axes(&plan.order_b).reshape(Distribution::cyclic(
            vec![plan.k, plan.n, plan.batches],
            processes,
        ));
        let c = self
            .permute_axes(&plan.order_c)
            .reshape(Distribution::cyclic(
                vec![plan.m, plan.n, plan.batches],
                processes,
            ));
        let mut folded = Self::new(
            self.context(),
            Distribution::cyclic(vec![plan.m, plan.n, plan.batches], processes),
            self.algebra().clone(),
        );
        let matrix_len = plan.m.checked_mul(plan.n).unwrap();
        for batch in 0..plan.batches {
            let aa = a
                .slice(&[0..plan.m, 0..plan.k, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.k], processes));
            let bb = b
                .slice(&[0..plan.k, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.k, plan.n], processes));
            let mut cc = c
                .slice(&[0..plan.m, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.n], processes));
            cc.gemm_sparse_dense(&aa, &bb, grid, alpha.clone(), beta.clone());
            let pairs: Vec<_> = cc
                .local_pairs()
                .into_iter()
                .filter(|(key, _)| cc.distribution().owner(*key) == self.context().rank())
                .map(|(key, value)| (key + batch * matrix_len, value))
                .collect();
            folded.write_add(&pairs);
        }

        let mut restored = folded
            .reshape(Distribution::cyclic(plan.canonical_c_shape, processes))
            .permute_axes(&plan.inverse_c);
        restored.redistribute(original);
        *self = restored;
        Ok(())
    }

    /// Contract dense A and sparse B into a dense fully folded output.
    pub fn contract_from_dense_sparse(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &SparseTensor<'_, '_, A>,
        indices_b: &str,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
    ) -> Result<(), Rejected> {
        if let Some([_, _, output]) = repeated_projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c])? {
            assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
            validate_grid(grid, self.context().size());
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_from_dense_sparse(&ic, &aa, &ia, &bb, &ib, grid, alpha, beta)?;
            if output.repeated() { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return Ok(());
        }
        let plan = Plan::new(
            [
                &a.distribution().shape,
                &b.distribution().shape,
                &self.distribution().shape,
            ],
            [indices_a, indices_b, indices_c],
        )?;
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        validate_grid(grid, self.context().size());

        let processes = self.context().size();
        let original = self.distribution().clone();
        let a = a.permute_axes(&plan.order_a).reshape(Distribution::cyclic(
            vec![plan.m, plan.k, plan.batches],
            processes,
        ));
        let b = b.permute_axes(&plan.order_b).reshape(Distribution::cyclic(
            vec![plan.k, plan.n, plan.batches],
            processes,
        ));
        let c = self
            .permute_axes(&plan.order_c)
            .reshape(Distribution::cyclic(
                vec![plan.m, plan.n, plan.batches],
                processes,
            ));
        let mut folded = Self::new(
            self.context(),
            Distribution::cyclic(vec![plan.m, plan.n, plan.batches], processes),
            self.algebra().clone(),
        );
        let matrix_len = plan.m.checked_mul(plan.n).unwrap();
        for batch in 0..plan.batches {
            let aa = a
                .slice(&[0..plan.m, 0..plan.k, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.k], processes));
            let bb = b
                .slice(&[0..plan.k, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.k, plan.n], processes));
            let mut cc = c
                .slice(&[0..plan.m, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.n], processes));
            cc.gemm_dense_sparse(&aa, &bb, grid, alpha.clone(), beta.clone());
            let pairs: Vec<_> = cc
                .local_pairs()
                .into_iter()
                .filter(|(key, _)| cc.distribution().owner(*key) == self.context().rank())
                .map(|(key, value)| (key + batch * matrix_len, value))
                .collect();
            folded.write_add(&pairs);
        }

        let mut restored = folded
            .reshape(Distribution::cyclic(plan.canonical_c_shape, processes))
            .permute_axes(&plan.inverse_c);
        restored.redistribute(original);
        *self = restored;
        Ok(())
    }
}
