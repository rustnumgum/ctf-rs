//! Offset-only reshuffle plans for compressed symmetric storage.
//!
//! Canonical coordinates are generated directly from symmetry bounds; the
//! rectangular tensor domain is never expanded or filtered after the fact.

use crate::{
    mapping::Distribution,
    symmetry::Symmetry,
    symmetric_distribution::SymmetricDistribution,
};

/// Visit canonical packed keys in ascending global-key order. Axis zero is
/// the fastest coordinate, so recursion starts at the highest axis and each
/// lower axis is bounded by its next coordinate in the same symmetry group.
pub(crate) fn visit_canonical_keys(
    distribution: &SymmetricDistribution,
    rank: usize,
    mut visit: impl FnMut(usize),
) {
    let rectangular = distribution.distribution();
    if rectangular.shape.iter().any(|&extent| extent == 0) {
        return;
    }
    if rectangular.shape.is_empty() {
        visit(0);
        return;
    }

    let rank_coordinates = rectangular.topology.coordinates(rank);
    let residues: Vec<_> = rectangular
        .mappings
        .iter()
        .map(|mapping| mapping.physical_rank(&rank_coordinates))
        .collect();
    if residues
        .iter()
        .zip(&rectangular.shape)
        .any(|(&residue, &extent)| residue >= extent)
    {
        return;
    }

    let phases: Vec<_> = rectangular
        .mappings
        .iter()
        .map(|mapping| mapping.physical_phase())
        .collect();
    let links = distribution.links();
    let mut coordinates = residues.clone();

    fn visit_axis(
        axis: usize,
        rectangular: &Distribution,
        links: &[Symmetry],
        residues: &[usize],
        phases: &[usize],
        coordinates: &mut [usize],
        visit: &mut impl FnMut(usize),
    ) {
        let upper = if links[axis] == Symmetry::NS {
            rectangular.shape[axis]
        } else {
            let next = coordinates[axis + 1];
            next + usize::from(links[axis] == Symmetry::SY)
        };
        let mut coordinate = residues[axis];
        while coordinate < upper {
            coordinates[axis] = coordinate;
            if axis == 0 {
                visit(rectangular.encode_key(coordinates));
            } else {
                visit_axis(
                    axis - 1,
                    rectangular,
                    links,
                    residues,
                    phases,
                    coordinates,
                    visit,
                );
            }
            coordinate += phases[axis];
        }
        coordinates[axis] = residues[axis];
    }

    visit_axis(
        rectangular.shape.len() - 1,
        rectangular,
        links,
        &residues,
        &phases,
        &mut coordinates,
        &mut visit,
    );
}
