// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! ROR traversal and value-only pack/unpack for dense DGTOG buckets.

use crate::{algebra::Wire, mapping::Distribution};

fn visit_coordinates(
    shape: &[usize],
    phase: &[usize],
    residue: &[usize],
    axis: usize,
    coordinates: &mut [usize],
    visit: &mut impl FnMut(&[usize]),
) {
    if axis == shape.len() {
        visit(coordinates);
        return;
    }
    let coordinate_axis = shape.len() - 1 - axis;
    let mut coordinate = residue[coordinate_axis];
    while coordinate < shape[coordinate_axis] {
        coordinates[coordinate_axis] = coordinate;
        visit_coordinates(shape, phase, residue, axis + 1, coordinates, visit);
        coordinate += phase[coordinate_axis];
    }
}

/// Source ROR order: within one LCM replica bucket, dimension zero varies
/// fastest. The returned offsets exclude padding but retain virtual-block
/// placement in the rank-local array.
pub(crate) fn offsets(
    distribution: &Distribution,
    rank: usize,
    common_phase: &[usize],
    residues: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    assert_eq!(common_phase.len(), distribution.shape.len());
    residues
        .iter()
        .map(|residue| {
            assert_eq!(residue.len(), distribution.shape.len());
            let mut bucket = Vec::new();
            let mut coordinates = vec![0; distribution.shape.len()];
            visit_coordinates(
                &distribution.shape,
                common_phase,
                residue,
                0,
                &mut coordinates,
                &mut |coordinates| {
                    let key = distribution.encode_key(coordinates);
                    bucket.push(distribution.local_offset(rank, key));
                },
            );
            bucket
        })
        .collect()
}

pub(crate) fn pack<T: Wire>(
    data: &[T],
    offsets: &[Vec<usize>],
    counts: &[usize],
    displacements: &[usize],
) -> Vec<u8> {
    assert_eq!(offsets.len(), counts.len());
    assert_eq!(offsets.len(), displacements.len());
    let elements = counts
        .last()
        .zip(displacements.last())
        .map_or(0, |(&count, &displacement)| count + displacement);
    let mut packed = Vec::with_capacity(elements * T::WIDTH);
    for (((bucket, &count), &displacement), expected_bucket) in offsets
        .iter()
        .zip(counts)
        .zip(displacements)
        .zip(0..)
    {
        assert_eq!(bucket.len(), count, "DGTOG bucket {expected_bucket} count");
        assert_eq!(packed.len(), displacement * T::WIDTH);
        for &offset in bucket {
            data[offset].encode(&mut packed);
        }
    }
    assert_eq!(packed.len(), elements * T::WIDTH);
    packed
}

pub(crate) fn unpack<T: Wire>(
    packed: &[u8],
    offsets: &[Vec<usize>],
    counts: &[usize],
    displacements: &[usize],
    data: &mut [T],
) {
    assert_eq!(offsets.len(), counts.len());
    assert_eq!(offsets.len(), displacements.len());
    for (bucket, (&count, &displacement)) in
        offsets.iter().zip(counts.iter().zip(displacements))
    {
        assert_eq!(bucket.len(), count);
        let begin = displacement * T::WIDTH;
        let end = begin + count * T::WIDTH;
        for (&offset, value) in bucket.iter().zip(packed[begin..end].chunks_exact(T::WIDTH)) {
            data[offset] = T::decode(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{offsets, pack, unpack};
    use crate::mapping::{Distribution, Mapping, Topology};

    #[test]
    fn ror_offsets_cross_virtual_blocks_and_skip_padding() {
        let topology = Topology::new(vec![2]);
        let mut mapping = Mapping::Unmapped;
        mapping.augment_physical(&topology, 0);
        mapping.augment_virtual(6);
        let distribution = Distribution::new(vec![8, 2], topology, vec![mapping, Mapping::Unmapped]);
        let residues = vec![vec![0, 0], vec![2, 0], vec![4, 0]];
        let bucket_offsets = offsets(&distribution, 0, &[6, 1], &residues);
        assert_eq!(bucket_offsets, vec![vec![0, 1, 2, 3], vec![4, 6], vec![8, 10]]);

        let data: Vec<u64> = (0..distribution.local_len() as u64).collect();
        let counts: Vec<_> = bucket_offsets.iter().map(Vec::len).collect();
        let displacements = vec![0, 4, 6];
        let packed = pack(&data, &bucket_offsets, &counts, &displacements);
        let mut output = vec![u64::MAX; data.len()];
        unpack(&packed, &bucket_offsets, &counts, &displacements, &mut output);
        for bucket in bucket_offsets {
            for offset in bucket {
                assert_eq!(output[offset], data[offset]);
            }
        }
        assert_eq!(output[5], u64::MAX);
    }
}
