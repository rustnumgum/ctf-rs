// Adapted from cc4s CTF interface/matrix.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Distributed native matrix operations, following interface/matrix.cxx's
//! read_mat -> ScaLAPACK -> tensor/get_tri sequence. No global tensor gather.
use crate::{
    algebra::Arithmetic,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

fn distribution(shape: &[usize], grid: [usize; 2]) -> Distribution {
    let topology = Topology::new(grid.to_vec());
    let mut row = Mapping::Unmapped;
    row.augment_physical(&topology, 0);
    let mut column = Mapping::Unmapped;
    column.augment_physical(&topology, 1);
    Distribution::new(shape.to_vec(), topology, vec![row, column])
}
impl<'c, 'r> Tensor<'c, 'r, Arithmetic<f64>> {
    /// Collective thin QR: Q has shape m*min(m,n), R has min(m,n)*n.
    /// Householder QR and explicit Q generation execute in ScaLAPACK; output
    /// tensors retain the selected grid and are not gathered to a single rank.
    pub fn qr(&self, grid: [usize; 2]) -> Result<(Self, Self), i32> {
        assert_eq!(self.distribution().shape.len(), 2);
        let (m, n) = (self.distribution().shape[0], self.distribution().shape[1]);
        let k = m.min(n);
        let mut source = self.clone();
        source.redistribute(distribution(&[m, n], grid));
        let blacs = self.context().inner.scalapack_grid(grid[0], grid[1]);
        let operation = (|| {
            let desc = blacs.descriptor(m, n, 1, 1, m.div_ceil(grid[0]).max(1))?;
            let mut input = source.local_storage().to_vec();
            input.resize(
                (m.div_ceil(grid[0]).max(1) * n.div_ceil(grid[1])).max(1),
                0.,
            );
            let (q_values, r_values) = blacs.qr(m, n, &input, &desc)?;
            let mut q = Self::new(
                self.context(),
                distribution(&[m, k], grid),
                Arithmetic::new(),
            );
            let mut r = Self::new(
                self.context(),
                distribution(&[k, n], grid),
                Arithmetic::new(),
            );
            let dist = source.distribution();
            let rank = self.context().rank();
            q.transform(|key, value| *value = q_values[dist.local_offset(rank, key)]);
            r.transform(|key, value| {
                let row = key % k;
                let col = key / k;
                *value = if row <= col {
                    r_values[dist.local_offset(rank, row + col * m)]
                } else {
                    0.
                };
            });
            Ok((q, r))
        })();
        blacs.close();
        operation
    }
    /// Collective thin SVD returning distributed U, singular-value vector, VT.
    /// Native replicated singular values are assigned directly to their owners;
    /// matrix factors stay distributed throughout PDGESVD and reconstruction.
    pub fn svd(&self, grid: [usize; 2]) -> Result<(Self, Self, Self), i32> {
        assert_eq!(self.distribution().shape.len(), 2);
        let (m, n) = (self.distribution().shape[0], self.distribution().shape[1]);
        let k = m.min(n);
        let mut source = self.clone();
        source.redistribute(distribution(&[m, n], grid));
        let mut u = Self::new(
            self.context(),
            distribution(&[m, k], grid),
            Arithmetic::new(),
        );
        let mut vt = Self::new(
            self.context(),
            distribution(&[k, n], grid),
            Arithmetic::new(),
        );
        let mut singular = Self::new(
            self.context(),
            Distribution::cyclic(vec![k], self.context().size()),
            Arithmetic::new(),
        );
        let blacs = self.context().inner.scalapack_grid(grid[0], grid[1]);
        let operation: Result<(), i32> = (|| {
            let lda = m.div_ceil(grid[0]).max(1);
            let ldvt = k.div_ceil(grid[0]).max(1);
            let da = blacs.descriptor(m, n, 1, 1, lda)?;
            let du = blacs.descriptor(m, k, 1, 1, lda)?;
            let dvt = blacs.descriptor(k, n, 1, 1, ldvt)?;
            let mut input = source.local_storage().to_vec();
            input.resize((lda * n.div_ceil(grid[1])).max(1), 0.);
            let mut uv = vec![0.; (lda * k.div_ceil(grid[1])).max(1)];
            let mut vtv = vec![0.; (ldvt * n.div_ceil(grid[1])).max(1)];
            let values = blacs.svd(m, n, &input, &da, &mut uv, &du, &mut vtv, &dvt)?;
            let ud = u.distribution().clone();
            let vd = vt.distribution().clone();
            let rank = self.context().rank();
            u.transform(|key, value| *value = uv[ud.local_offset(rank, key)]);
            vt.transform(|key, value| *value = vtv[vd.local_offset(rank, key)]);
            singular.transform(|key, value| *value = values[key]);
            Ok(())
        })();
        blacs.close();
        operation?;
        Ok((u, singular, vt))
    }
    /// Collective PD POTRF on a caller-selected grid with cyclic block size 1.
    /// Returns only the requested triangle in the input's original distribution.
    /// The BLACS grid is explicitly closed before returning, including info errors.
    pub fn cholesky(&self, grid: [usize; 2], lower: bool) -> Result<Self, i32> {
        assert_eq!(self.distribution().shape.len(), 2);
        let n = self.distribution().shape[0];
        assert_eq!(self.distribution().shape[1], n);
        let mut result = self.clone();
        result.redistribute(distribution(&[n, n], grid));
        let blacs = self.context().inner.scalapack_grid(grid[0], grid[1]);
        let operation: Result<(), i32> = (|| {
            let desc = blacs.descriptor(n, n, 1, 1, n.div_ceil(grid[0]).max(1))?;
            let mut values = result.local_storage().to_vec();
            values.resize(values.len().max(1), 0.);
            blacs.cholesky(n, &mut values, &desc, lower)?;
            let dist = result.distribution().clone();
            let rank = self.context().rank();
            result.transform(|key, value| {
                let row = key % n;
                let col = key / n;
                *value = if (lower && row >= col) || (!lower && row <= col) {
                    values[dist.local_offset(rank, key)]
                } else {
                    0.
                };
            });
            Ok(())
        })();
        blacs.close();
        operation?;
        result.redistribute(self.distribution().clone());
        Ok(result)
    }
    /// Solve op(T)*X=B or X*op(T)=B using PDTRSM, keeping input tensors unchanged.
    /// `self` is B; diagonal entries of T are non-unit, as in upstream solve_tri.
    pub fn solve_tri(
        &self,
        factor: &Self,
        grid: [usize; 2],
        lower: bool,
        from_left: bool,
        transpose: bool,
    ) -> Result<Self, i32> {
        assert!(std::ptr::eq(self.context(), factor.context()));
        assert_eq!(self.distribution().shape.len(), 2);
        assert_eq!(factor.distribution().shape.len(), 2);
        let (m, n) = (self.distribution().shape[0], self.distribution().shape[1]);
        let order = if from_left { m } else { n };
        assert_eq!(factor.distribution().shape, vec![order, order]);
        let mut a = factor.clone();
        a.redistribute(distribution(&[order, order], grid));
        let mut result = self.clone();
        result.redistribute(distribution(&[m, n], grid));
        let blacs = self.context().inner.scalapack_grid(grid[0], grid[1]);
        let operation: Result<(), i32> = (|| {
            let da = blacs.descriptor(order, order, 1, 1, order.div_ceil(grid[0]).max(1))?;
            let db = blacs.descriptor(m, n, 1, 1, m.div_ceil(grid[0]).max(1))?;
            let mut av = a.local_storage().to_vec();
            av.resize(av.len().max(1), 0.);
            let mut values = result.local_storage().to_vec();
            values.resize(
                (m.div_ceil(grid[0]).max(1) * n.div_ceil(grid[1])).max(1),
                0.,
            );
            blacs.solve_tri(
                m,
                n,
                &av,
                &da,
                &mut values,
                &db,
                lower,
                from_left,
                transpose,
            );
            let dist = result.distribution().clone();
            let rank = self.context().rank();
            result.transform(|key, value| *value = values[dist.local_offset(rank, key)]);
            Ok(())
        })();
        blacs.close();
        operation?;
        result.redistribute(self.distribution().clone());
        Ok(result)
    }
}
