// Adapted from cc4s CTF interface/matrix.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Distributed native matrix operations, following interface/matrix.cxx's
//! read_mat -> ScaLAPACK -> tensor/get_tri sequence. No global tensor gather.
use crate::{
    algebra::{Arithmetic, Wire},
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
    /// Symmetric eigensolve using the source's largest-square-grid subworld
    /// strategy. Reads the upper triangle; returns (eigenvectors, eigenvalues).
    /// For non-square process counts, only the first floor(sqrt(np))^2 ranks
    /// compute, then factors are distributed back to the original context.
    pub fn eigh(&self) -> Result<(Self, Self), i32> {
        assert_eq!(self.distribution().shape.len(), 2);
        let n = self.distribution().shape[0];
        assert_eq!(self.distribution().shape[1], n);
        let context = self.context();
        let mut side = 1;
        while (side + 1) * (side + 1) <= context.size() {
            side += 1;
        }
        let active = side * side;
        let mapped = distribution(&[n, n], [side, side]);
        // add_to_subworld: only canonical source owners send, directly to the
        // rank that owns each entry in the child grid. No root tensor assembly.
        let mut buckets = vec![Vec::new(); context.size()];
        for (key, value) in self.local_pairs() {
            if self.distribution().owner(key) != context.rank() {
                continue;
            }
            let destination = mapped.owner(key);
            (key as u64).encode(&mut buckets[destination]);
            value.encode(&mut buckets[destination]);
        }
        let incoming = context.inner.exchange(&buckets);
        let child = context.split(
            if context.rank() < active {
                Some(1)
            } else {
                None
            },
            context.rank() as i32,
        );
        let mut vector_pairs = Vec::new();
        let mut value_pairs = Vec::new();
        let mut info = 0;
        if let Some(child) = child {
            let mut input = vec![0.; mapped.local_len().max(1)];
            for bytes in incoming {
                for pair in bytes.chunks_exact(16) {
                    let key = u64::decode(&pair[..8]) as usize;
                    input[mapped.local_offset(child.rank(), key)] = f64::decode(&pair[8..]);
                }
            }
            let grid = child.inner.scalapack_grid(side, side);
            let result = (|| {
                let desc = grid.descriptor(n, n, 1, 1, n.div_ceil(side).max(1))?;
                let mut vectors = vec![0.; input.len()];
                let values = grid.eigh(n, &input, &desc, &mut vectors)?;
                for (offset, &value) in vectors.iter().enumerate() {
                    if let Some(key) = mapped.global_key(child.rank(), offset) {
                        vector_pairs.push((key, value));
                    }
                }
                if child.rank() == 0 {
                    value_pairs.extend(values.into_iter().enumerate());
                }
                Ok::<(), i32>(())
            })();
            grid.close();
            if let Err(error) = result {
                info = error;
            }
            child.close();
        }
        let statuses = context.inner.all_gather_i32(info);
        if let Some(error) = statuses.into_iter().find(|&value| value != 0) {
            return Err(error);
        }
        // add_from_subworld: all parent ranks participate, including ranks that
        // did not enter ScaLAPACK; writes restore each output's distribution.
        let mut vectors = Self::new(context, self.distribution().clone(), Arithmetic::new());
        vectors.write_add(&vector_pairs);
        let mut values = Self::new(
            context,
            Distribution::cyclic(vec![n], context.size()),
            Arithmetic::new(),
        );
        values.write_add(&value_pairs);
        Ok((vectors, values))
    }
    /// Source rank/threshold postprocessing: keep singular values >= threshold.
    /// A computed zero rank returns the full factors in this pinned version.
    pub fn svd_truncated(
        &self,
        grid: [usize; 2],
        rank: Option<usize>,
        threshold: f64,
    ) -> Result<(Self, Self, Self), i32> {
        let (u, s, vt) = self.svd(grid)?;
        let k = s.distribution().shape[0];
        let mut retained = rank.unwrap_or(k);
        if threshold > 0. {
            let values = s.read(&(0..k).collect::<Vec<_>>());
            let cutoff = values.partition_point(|value| value.abs() >= threshold);
            retained = if retained > 0 {
                retained.min(cutoff)
            } else {
                cutoff
            };
        }
        if retained > 0 && retained < k {
            let m = u.distribution().shape[0];
            let n = vt.distribution().shape[1];
            Ok((
                u.slice(&[0..m, 0..retained]),
                s.slice(&[0..retained]),
                vt.slice(&[0..retained, 0..n]),
            ))
        } else {
            Ok((u, s, vt))
        }
    }
    /// Source svd_rand: initialize/QR (unless a guess is supplied), apply A*A^T
    /// and QR, retain requested columns, SVD U^T*A, and rotate the left factor.
    /// All matrix intermediates remain distributed; seed is explicitly supplied.
    pub fn svd_randomized(
        &self,
        grid: [usize; 2],
        rank: usize,
        iterations: usize,
        oversampling: usize,
        seed: u64,
        guess: Option<&Self>,
    ) -> Result<(Self, Self, Self), i32> {
        assert_eq!(self.distribution().shape.len(), 2);
        let (m, n) = (self.distribution().shape[0], self.distribution().shape[1]);
        assert!(rank > 0 && rank <= m.min(n));
        let width = (rank + oversampling).min(m.min(n));
        let mut subspace = if let Some(guess) = guess {
            assert!(std::ptr::eq(self.context(), guess.context()));
            assert!(rank + oversampling <= m.min(n));
            assert_eq!(guess.distribution().shape, vec![m, width]);
            guess.clone()
        } else {
            let mut values = Self::new(
                self.context(),
                distribution(&[m, width], grid),
                Arithmetic::new(),
            );
            let mut state = ((seed.wrapping_add(self.context().rank() as u64)) << 16) | 0x330e;
            values.transform(|_, value| {
                state = (state.wrapping_mul(0x5deece66d).wrapping_add(11)) & ((1 << 48) - 1);
                *value = 2. * state as f64 / (1u64 << 48) as f64 - 1.;
            });
            values.qr(grid)?.0
        };
        for _ in 0..iterations {
            let transpose = self.permute_axes(&[1, 0]);
            let mut gram = Self::new(
                self.context(),
                distribution(&[m, m], grid),
                Arithmetic::new(),
            );
            gram.gemm_2d::<crate::linalg::Native>(self, &transpose, grid, 1., 0.);
            let mut next = Self::new(
                self.context(),
                distribution(&[m, width], grid),
                Arithmetic::new(),
            );
            next.gemm_2d::<crate::linalg::Native>(&gram, &subspace, grid, 1., 0.);
            subspace = next.qr(grid)?.0;
        }
        let u = if width > rank {
            subspace.slice(&[0..m, 0..rank])
        } else {
            subspace
        };
        let transpose = u.permute_axes(&[1, 0]);
        let mut projected = Self::new(
            self.context(),
            distribution(&[rank, n], grid),
            Arithmetic::new(),
        );
        projected.gemm_2d::<crate::linalg::Native>(&transpose, self, grid, 1., 0.);
        let (rotation, s, vt) = projected.svd(grid)?;
        let mut left = Self::new(
            self.context(),
            distribution(&[m, rank], grid),
            Arithmetic::new(),
        );
        left.gemm_2d::<crate::linalg::Native>(&u, &rotation, grid, 1., 0.);
        Ok((left, s, vt))
    }
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
