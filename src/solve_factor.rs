// Adapted from cc4s CTF interface/multilinear.cxx Solve_Factor.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::Arithmetic, mapping::{Distribution, Mapping}, tensor::Tensor};
use super::physical_mapping;

impl<'c, 'r> Tensor<'c, 'r, Arithmetic<f64>> {
    /// Solve weighted normal equations for one factor, using auxiliary-first
    /// matrices [rank, mode_length]. `self` contains tensor weights. Supply
    /// factors in ascending mode order, excluding the output mode.
    /// Gram systems are reduce-scattered over complementary MPI fibers and
    /// solved locally with POSV, not gathered for a root-only factorization.
    pub fn solve_factor(&self, output_mode: usize, factors: &[&Self], rhs: &Self)
        -> Result<Self, i32> {
        let dist = self.distribution();
        let order = dist.shape.len();
        assert!(order >= 2 && output_mode < order);
        assert_eq!(factors.len(), order - 1);
        assert!(std::ptr::eq(self.context(), rhs.context()));
        assert_eq!(rhs.distribution().shape.len(), 2);
        let k = rhs.distribution().shape[0];
        assert!(k > 0 && dist.shape[output_mode] > 0);
        assert_eq!(rhs.distribution().shape[1], dist.shape[output_mode]);
        let rank = self.context().rank();
        let coordinates = dist.topology.coordinates(rank);
        let mappings: Vec<_> = dist.mappings.iter().map(physical_mapping).collect();
        let phases: Vec<_> = mappings.iter().map(Mapping::physical_phase).collect();
        let color = mappings[output_mode].physical_rank(&coordinates);
        let slice = self.context().split(Some(color as i32), rank as i32).unwrap();
        let local_rows = dist.shape[output_mode].div_ceil(phases[output_mode]);
        let rows_per_worker = local_rows.div_ceil(slice.size());
        let padded_rows = rows_per_worker * slice.size();
        let mut arrays = Vec::with_capacity(order);
        let mut factor_index = 0;
        for mode in 0..order {
            if mode == output_mode {
                arrays.push(Vec::new());
                continue;
            }
            let factor = factors[factor_index];
            factor_index += 1;
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert_eq!(factor.distribution().shape, vec![k, dist.shape[mode]]);
            let mapped = Distribution::new(vec![k, dist.shape[mode]], dist.topology.clone(),
                vec![Mapping::Unmapped, mappings[mode].clone()]);
            let fiber = self.context().split(
                Some(mappings[mode].physical_rank(&coordinates) as i32), rank as i32).unwrap();
            let keys: Vec<_> = if fiber.rank() == 0 {
                (0..mapped.local_len()).filter_map(|offset| mapped.global_key(rank, offset)).collect()
            } else { Vec::new() };
            let mut values = vec![0.; mapped.local_len()];
            for (key, value) in keys.iter().zip(factor.read(&keys)) {
                values[mapped.local_offset(rank, *key)] = value;
            }
            fiber.broadcast(0, &mut values);
            fiber.close();
            arrays.push(values);
        }
        let mut grams = vec![0.; padded_rows * k * k];
        let mut row = vec![1.; k];
        for (key, weight) in self.local_pairs() {
            if dist.owner(key) != rank { continue; }
            let indices = dist.decode_key(key);
            row.fill(1.);
            for mode in 0..order {
                if mode == output_mode { continue; }
                let offset = (indices[mode] / phases[mode]) * k;
                for auxiliary in 0..k { row[auxiliary] *= arrays[mode][offset + auxiliary]; }
            }
            let offset = (indices[output_mode] / phases[output_mode]) * k * k;
            crate::ffi::factor::syr(k, weight, &row, &mut grams[offset..offset + k * k]);
        }
        let mut local_grams = slice.inner.reduce_scatter_f64(&grams, rows_per_worker * k * k);
        let rhs_mapped = Distribution::new(vec![k, dist.shape[output_mode]], dist.topology.clone(),
            vec![Mapping::Unmapped, mappings[output_mode].clone()]);
        let keys: Vec<_> = if slice.rank() == 0 {
            (0..rhs_mapped.local_len()).filter_map(|offset| rhs_mapped.global_key(rank, offset)).collect()
        } else { Vec::new() };
        // Explicitly pad the root RHS before equal-size Scatter: the source
        // otherwise reads past its buffer when rows do not divide slice size.
        let mut rhs_values = vec![0.; padded_rows * k];
        for (key, value) in keys.iter().zip(rhs.read(&keys)) {
            rhs_values[rhs_mapped.local_offset(rank, *key)] = value;
        }
        let mut local_rhs = slice.inner.scatter_f64(0, &rhs_values, rows_per_worker * k);
        let mut info = 0;
        for local_row in 0..rows_per_worker {
            let row_index = slice.rank() * rows_per_worker + local_row;
            let global_row = row_index * phases[output_mode] + color;
            if global_row >= dist.shape[output_mode] { continue; }
            let result = crate::ffi::factor::posv(k,
                &mut local_grams[local_row * k * k..(local_row + 1) * k * k],
                &mut local_rhs[local_row * k..(local_row + 1) * k]);
            if let Err(error) = result { info = error; break; }
        }
        let errors = self.context().inner.all_gather_i32(info);
        if let Some(error) = errors.into_iter().find(|&error| error != 0) {
            slice.close();
            return Err(error);
        }
        let solved = slice.inner.gather_f64(0, &local_rhs);
        let pairs: Vec<_> = if slice.rank() == 0 {
            (0..rhs_mapped.local_len()).filter_map(|offset|
                rhs_mapped.global_key(rank, offset).map(|key| (key, solved[offset]))).collect()
        } else { Vec::new() };
        slice.close();
        // Only reduced roots contribute: non-root RHS replicas must not be
        // added to the solved factor, unlike the source's stale replica path.
        let mut result = Self::new(self.context(), rhs.distribution().clone(), Arithmetic::new());
        result.write_add(&pairs);
        Ok(result)
    }
}
