//! Dense nonsymmetric slice extraction and insertion.
//!
//! A distributed extraction first copies the rank-local intersection and then
//! performs the cyclic owner shift induced by the lower slice corner.  The
//! resulting allocation retains the input mappings and contains no gathered
//! global tensor.

use std::ops::Range;

use crate::{
    algebra::{Monoid, Semiring, Wire},
    context::Context,
    mapping::Distribution,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlicePlan {
    pub rank: usize,
    pub output_distribution: Distribution,
    pub send_rank: usize,
    pub receive_rank: usize,
    /// `(input offset, output offset before the rank shift)`.
    pub copies: Vec<(usize, usize)>,
}

impl SlicePlan {
    pub fn new(distribution: &Distribution, rank: usize, ranges: &[Range<usize>]) -> Self {
        assert_eq!(ranges.len(), distribution.shape.len());
        assert!(rank < distribution.topology.size());
        for (range, &extent) in ranges.iter().zip(&distribution.shape) {
            assert!(range.start <= range.end && range.end <= extent);
        }

        let offsets: Vec<_> = ranges.iter().map(|range| range.start).collect();
        let mut output_distribution = distribution.clone();
        output_distribution.shape = ranges
            .iter()
            .map(|range| range.end - range.start)
            .collect();
        let send_rank = distribution.shifted_rank(rank, &offsets, false);
        let receive_rank = distribution.shifted_rank(rank, &offsets, true);

        let mut copies = Vec::new();
        for input_offset in 0..distribution.local_len() {
            let Some(key) = distribution.global_key(rank, input_offset) else {
                continue;
            };
            let coordinates = distribution.decode_key(key);
            if !coordinates
                .iter()
                .zip(ranges)
                .all(|(&coordinate, range)| range.contains(&coordinate))
            {
                continue;
            }
            let sliced: Vec<_> = coordinates
                .iter()
                .zip(&offsets)
                .map(|(&coordinate, &offset)| coordinate - offset)
                .collect();
            let output_key = output_distribution.encode_key(&sliced);
            copies.push((
                input_offset,
                output_distribution.local_offset(send_rank, output_key),
            ));
        }

        Self {
            rank,
            output_distribution,
            send_rank,
            receive_rank,
            copies,
        }
    }

    pub fn extract_local<A: Monoid>(&self, algebra: &A, input: &[A::Element]) -> Vec<A::Element> {
        let mut output = vec![algebra.zero(); self.output_distribution.local_len()];
        for &(input_offset, output_offset) in &self.copies {
            output[output_offset] = input[input_offset].clone();
        }
        output
    }

    pub fn execute<A: Monoid>(
        &self,
        context: &Context<'_>,
        algebra: &A,
        input: &[A::Element],
    ) -> Vec<A::Element>
    where
        A::Element: Wire,
    {
        assert_eq!(context.size(), self.output_distribution.topology.size());
        assert_eq!(context.rank(), self.rank);
        let mut output = self.extract_local(algebra, input);
        if self.send_rank == context.rank() {
            return output;
        }

        let mut send = Vec::with_capacity(output.len() * A::Element::WIDTH);
        for value in &output {
            value.encode(&mut send);
        }
        let mut received = vec![0; send.len()];
        if !send.is_empty() {
            context
                .inner
                .send_receive(&send, self.send_rank, self.receive_rank, &mut received);
        }
        for (value, bytes) in output
            .iter_mut()
            .zip(received.chunks_exact(A::Element::WIDTH))
        {
            *value = A::Element::decode(bytes);
        }
        output
    }
}

/// `destination[ranges] = beta*destination[ranges] + alpha*source` in
/// column-major order.  Beta is not applied outside the selected region.
pub fn accumulate_local_slice<A: Semiring>(
    algebra: &A,
    destination_shape: &[usize],
    ranges: &[Range<usize>],
    source: &[A::Element],
    destination: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
) {
    assert_eq!(destination_shape.len(), ranges.len());
    assert_eq!(destination.len(), destination_shape.iter().product());
    for (range, &extent) in ranges.iter().zip(destination_shape) {
        assert!(range.start <= range.end && range.end <= extent);
    }
    let source_shape: Vec<_> = ranges
        .iter()
        .map(|range| range.end - range.start)
        .collect();
    assert_eq!(source.len(), source_shape.iter().product());
    let destination_strides = column_major_strides(destination_shape);

    for (source_offset, source_value) in source.iter().enumerate() {
        let mut remainder = source_offset;
        let destination_offset: usize = source_shape
            .iter()
            .zip(ranges)
            .zip(&destination_strides)
            .map(|((&extent, range), &stride)| {
                let coordinate = remainder % extent;
                remainder /= extent;
                (coordinate + range.start) * stride
            })
            .sum();
        destination[destination_offset] = algebra.add(
            &algebra.multiply(beta, &destination[destination_offset]),
            &algebra.multiply(alpha, source_value),
        );
    }
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape
        .iter()
        .map(|&extent| {
            let current = stride;
            stride *= extent;
            current
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{accumulate_local_slice, SlicePlan};
    use crate::{
        algebra::Arithmetic,
        mapping::{Distribution, Mapping, Topology},
    };

    #[test]
    fn local_slice_applies_alpha_beta_only_inside_destination_region() {
        let mut destination: Vec<i64> = (0..20).collect();
        let outside = destination.clone();
        accumulate_local_slice(
            &Arithmetic::<i64>::new(),
            &[4, 5],
            &[1..3, 2..5],
            &[10, 11, 12, 13, 14, 15],
            &mut destination,
            &2,
            &3,
        );
        for j in 0..5 {
            for i in 0..4 {
                let offset = i + 4 * j;
                if (1..3).contains(&i) && (2..5).contains(&j) {
                    let source = (i - 1) + 2 * (j - 2);
                    assert_eq!(destination[offset], 3 * outside[offset] + 2 * (10 + source as i64));
                } else {
                    assert_eq!(destination[offset], outside[offset]);
                }
            }
        }
    }

    #[test]
    fn extraction_metadata_preserves_virtual_offsets_and_cyclic_rank_shift() {
        let topology = Topology::new(vec![4]);
        let mut axis = Mapping::Unmapped;
        axis.augment_physical(&topology, 0);
        axis.augment_virtual(8);
        let distribution = Distribution::new(
            vec![9, 3],
            topology,
            vec![axis, Mapping::Unmapped],
        );
        for rank in 0..4 {
            let plan = SlicePlan::new(&distribution, rank, &[3..8, 1..3]);
            assert_eq!(plan.output_distribution.shape, [5, 2]);
            assert_eq!(plan.send_rank, (rank + 1) % 4);
            assert_eq!(plan.receive_rank, (rank + 3) % 4);
            for &(old_offset, new_offset) in &plan.copies {
                let old_key = distribution.global_key(rank, old_offset).unwrap();
                let old = distribution.decode_key(old_key);
                let new_key = plan
                    .output_distribution
                    .global_key(plan.send_rank, new_offset)
                    .unwrap();
                let new = plan.output_distribution.decode_key(new_key);
                assert_eq!(new, [old[0] - 3, old[1] - 1]);
            }
        }
    }
}
