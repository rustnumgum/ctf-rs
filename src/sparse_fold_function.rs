// Adapted from cc4s CTF contraction/contraction.cxx sparse folding and
// interface/functions.h Bivar_Function folded kernels at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Fully folded custom bivariate sparse contractions.

use super::{Plan, validate_grid, validated_projections};
use crate::{
    algebra::{Semiring, Wire},
    mapping::Distribution,
    sparse::SparseTensor,
    tensor::Tensor,
};

impl<'c, 'r, A: Semiring + Clone> SparseTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Contract sparse A and B into sparse C with a source-compatible folded
    /// custom function. Arbitrary functions are not distributively pre-reduced.
    pub fn contract_from_function(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
        function: impl Fn(&A::Element, &A::Element) -> A::Element,
    ) -> Result<(), crate::folding::Rejected> {
        let projections = validated_projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c],
        )?;
        let plan = Plan::new(
            [&projections[0].shape, &projections[1].shape, &projections[2].shape],
            [&projections[0].labels, &projections[1].labels, &projections[2].labels],
        )?;
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        validate_grid(grid, self.context().size());
        assert!(alpha == self.algebra().one(),
            "source custom CSR kernel requires identity alpha");
        if projections.iter().any(|projection| projection.repeated()) {
            let output_repeated = projections[2].repeated();
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_from_function(&ic, &aa, &ia, &bb, &ib, grid,
                alpha, beta, function)?;
            if output_repeated { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return Ok(());
        }

        let processes = self.context().size();
        let original = self.distribution().clone();
        let a = a.permute_axes(&plan.order_a).reshape(Distribution::cyclic(
            vec![plan.m, plan.k, plan.batches], processes));
        let b = b.permute_axes(&plan.order_b).reshape(Distribution::cyclic(
            vec![plan.k, plan.n, plan.batches], processes));
        let c = self.permute_axes(&plan.order_c).reshape(Distribution::cyclic(
            vec![plan.m, plan.n, plan.batches], processes));
        let mut folded = Self::new(self.context(), Distribution::cyclic(
            vec![plan.m, plan.n, plan.batches], processes), self.algebra().clone());
        let matrix_len = plan.m.checked_mul(plan.n).unwrap();
        for batch in 0..plan.batches {
            let aa = a.slice(&[0..plan.m, 0..plan.k, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.k], processes));
            let bb = b.slice(&[0..plan.k, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.k, plan.n], processes));
            let mut cc = c.slice(&[0..plan.m, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.n], processes));
            cc.gemm_sparse_function(&aa, &bb, grid, alpha.clone(), beta.clone(), &function);
            let pairs: Vec<_> = cc.local_pairs().into_iter()
                .filter(|(key, _)| cc.distribution().owner(*key) == self.context().rank())
                .map(|(key, value)| (key + batch * matrix_len, value)).collect();
            folded.write_add(&pairs);
        }

        let mut restored = folded.reshape(Distribution::cyclic(
            plan.canonical_c_shape, processes)).permute_axes(&plan.inverse_c);
        restored.redistribute(original);
        *self = restored;
        Ok(())
    }
}

impl<'c, 'r, A: Semiring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Contract sparse A and B into dense C with a fully folded custom function.
    pub fn contract_from_sparse_function(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &SparseTensor<'_, '_, A>,
        indices_b: &str,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
        function: impl Fn(&A::Element, &A::Element) -> A::Element,
    ) -> Result<(), crate::folding::Rejected> {
        let projections = validated_projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c],
        )?;
        let plan = Plan::new(
            [&projections[0].shape, &projections[1].shape, &projections[2].shape],
            [&projections[0].labels, &projections[1].labels, &projections[2].labels],
        )?;
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        validate_grid(grid, self.context().size());
        assert!(alpha == self.algebra().one(),
            "source custom CSR kernel requires identity alpha");
        if projections.iter().any(|projection| projection.repeated()) {
            let output_repeated = projections[2].repeated();
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_from_sparse_function(&ic, &aa, &ia, &bb, &ib, grid,
                alpha, beta, function)?;
            if output_repeated { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return Ok(());
        }

        let processes = self.context().size();
        let original = self.distribution().clone();
        let a = a.permute_axes(&plan.order_a).reshape(Distribution::cyclic(
            vec![plan.m, plan.k, plan.batches], processes));
        let b = b.permute_axes(&plan.order_b).reshape(Distribution::cyclic(
            vec![plan.k, plan.n, plan.batches], processes));
        let c = self.permute_axes(&plan.order_c).reshape(Distribution::cyclic(
            vec![plan.m, plan.n, plan.batches], processes));
        let mut folded = Self::new(self.context(), Distribution::cyclic(
            vec![plan.m, plan.n, plan.batches], processes), self.algebra().clone());
        let matrix_len = plan.m.checked_mul(plan.n).unwrap();
        for batch in 0..plan.batches {
            let aa = a.slice(&[0..plan.m, 0..plan.k, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.k], processes));
            let bb = b.slice(&[0..plan.k, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.k, plan.n], processes));
            let mut cc = c.slice(&[0..plan.m, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.n], processes));
            cc.gemm_sparse_function(&aa, &bb, grid, alpha.clone(), beta.clone(), &function);
            let pairs: Vec<_> = cc.local_pairs().into_iter()
                .filter(|(key, _)| cc.distribution().owner(*key) == self.context().rank())
                .map(|(key, value)| (key + batch * matrix_len, value)).collect();
            folded.write_add(&pairs);
        }

        let mut restored = folded.reshape(Distribution::cyclic(
            plan.canonical_c_shape, processes)).permute_axes(&plan.inverse_c);
        restored.redistribute(original);
        *self = restored;
        Ok(())
    }

    /// Contract sparse A and dense B into dense C with a fully folded custom function.
    pub fn contract_from_sparse_dense_function(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        grid: [usize; 2],
        alpha: A::Element,
        beta: A::Element,
        function: impl Fn(&A::Element, &A::Element) -> A::Element,
    ) -> Result<(), crate::folding::Rejected> {
        let projections = validated_projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c],
        )?;
        let plan = Plan::new(
            [&projections[0].shape, &projections[1].shape, &projections[2].shape],
            [&projections[0].labels, &projections[1].labels, &projections[2].labels],
        )?;
        assert!(std::ptr::eq(self.context(), a.context()) && std::ptr::eq(self.context(), b.context()));
        validate_grid(grid, self.context().size());
        assert!(alpha == self.algebra().one(),
            "source custom CSR kernel requires identity alpha");
        if projections.iter().any(|projection| projection.repeated()) {
            let output_repeated = projections[2].repeated();
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_from_sparse_dense_function(&ic, &aa, &ia, &bb, &ib, grid,
                alpha, beta, function)?;
            if output_repeated { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return Ok(());
        }

        let processes = self.context().size();
        let original = self.distribution().clone();
        let a = a.permute_axes(&plan.order_a).reshape(Distribution::cyclic(
            vec![plan.m, plan.k, plan.batches], processes));
        let b = b.permute_axes(&plan.order_b).reshape(Distribution::cyclic(
            vec![plan.k, plan.n, plan.batches], processes));
        let c = self.permute_axes(&plan.order_c).reshape(Distribution::cyclic(
            vec![plan.m, plan.n, plan.batches], processes));
        let mut folded = Self::new(self.context(), Distribution::cyclic(
            vec![plan.m, plan.n, plan.batches], processes), self.algebra().clone());
        let matrix_len = plan.m.checked_mul(plan.n).unwrap();
        for batch in 0..plan.batches {
            let aa = a.slice(&[0..plan.m, 0..plan.k, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.k], processes));
            let bb = b.slice(&[0..plan.k, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.k, plan.n], processes));
            let mut cc = c.slice(&[0..plan.m, 0..plan.n, batch..batch + 1])
                .reshape(Distribution::cyclic(vec![plan.m, plan.n], processes));
            cc.gemm_sparse_dense_function(&aa, &bb, grid, alpha.clone(), beta.clone(), &function);
            let pairs: Vec<_> = cc.local_pairs().into_iter()
                .filter(|(key, _)| cc.distribution().owner(*key) == self.context().rank())
                .map(|(key, value)| (key + batch * matrix_len, value)).collect();
            folded.write_add(&pairs);
        }

        let mut restored = folded.reshape(Distribution::cyclic(
            plan.canonical_c_shape, processes)).permute_axes(&plan.inverse_c);
        restored.redistribute(original);
        *self = restored;
        Ok(())
    }
}
