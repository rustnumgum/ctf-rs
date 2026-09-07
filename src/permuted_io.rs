//! Parent/child dense permutation transfers adapted from
//! `tensor/untyped_tensor.cxx::permute`.

use crate::{
    algebra::{Semiring, Wire},
    mapping::Distribution,
    tensor::Tensor,
};

pub(crate) fn validate_maps(parent: &Distribution, child: &Distribution, maps: &[Vec<Option<usize>>]) {
    assert_eq!(maps.len(), child.shape.len());
    assert_eq!(maps.len(), parent.shape.len());
    for (axis, coordinates) in maps.iter().enumerate() {
        assert_eq!(coordinates.len(), child.shape[axis]);
        assert!(coordinates
            .iter()
            .all(|&coordinate| coordinate.is_none_or(|coordinate| coordinate < parent.shape[axis])));
    }
}

pub(crate) fn validate_parent_maps(parent: &Distribution, maps: &[Vec<Option<usize>>]) {
    assert_eq!(maps.len(), parent.shape.len());
    assert!(maps.iter().enumerate().all(|(axis, coordinates)| {
        coordinates
            .iter()
            .all(|&coordinate| coordinate.is_none_or(|coordinate| coordinate < parent.shape[axis]))
    }));
}

pub(crate) fn mapped_key(
    parent: &Distribution,
    child: &Distribution,
    maps: &[Vec<Option<usize>>],
    child_key: usize,
) -> Option<usize> {
    let coordinates = child.decode_key(child_key);
    let mut parent_coordinates = Vec::with_capacity(coordinates.len());
    for (axis, coordinate) in coordinates.into_iter().enumerate() {
        parent_coordinates.push(maps[axis][coordinate]?);
    }
    Some(parent.encode_key(&parent_coordinates))
}

impl<'c, 'r, A: Semiring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Gather parent values into an optionally present child tensor through
    /// per-axis child-to-parent coordinate maps. Parent ranks without a child
    /// still participate in the parent read with an empty request.
    pub fn gather_permuted_into(
        &self,
        destination: Option<&mut Tensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(destination
            .as_ref()
            .is_none_or(|destination| destination.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);

        for coordinates in maps {
            let retained: Vec<_> = coordinates.iter().flatten().copied().collect();
            assert_eq!(retained.iter().copied().collect::<std::collections::BTreeSet<_>>().len(),retained.len(),
                "gather coordinate maps must be injective");
        }

        let (parent_keys, child_keys) = if let Some(destination) = destination.as_ref() {
            validate_maps(self.distribution(), destination.distribution(), maps);
            let rank = destination.context().rank();
            let mut keys = Vec::new();
            for offset in 0..destination.data.len() {
                let Some(child_key) = destination.distribution().global_key(rank, offset) else {
                    continue;
                };
                if destination.distribution().owner(child_key) != rank {
                    continue;
                }
                if let Some(parent_key) = mapped_key(
                    self.distribution(),
                    destination.distribution(),
                    maps,
                    child_key,
                ) {
                    keys.push((parent_key, child_key));
                }
            }
            (
                keys.iter().map(|&(parent_key, _)| parent_key).collect(),
                keys.iter().map(|&(_, child_key)| child_key).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let values = self.read(&parent_keys);
        if let Some(destination) = destination {
            let pairs: Vec<_> = child_keys.into_iter().zip(values).collect();
            destination.write_scaled(&pairs, &alpha, &beta);
        }
    }

    /// Scatter nonzero child values into parent coordinates through per-axis
    /// child-to-parent maps. Unmapped coordinates and dense zeros contribute
    /// nothing, leaving untouched parent keys unchanged.
    pub fn scatter_permuted_from(
        &mut self,
        source: Option<&Tensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);

        let mut pairs = Vec::new();
        if let Some(source) = source {
            validate_maps(self.distribution(), source.distribution(), maps);
            let rank = source.context().rank();
            let zero = source.algebra().zero();
            for offset in 0..source.data.len() {
                let Some(child_key) = source.distribution().global_key(rank, offset) else {
                    continue;
                };
                if source.distribution().owner(child_key) != rank
                    || source.data[offset] == zero
                {
                    continue;
                }
                if let Some(parent_key) = mapped_key(
                    self.distribution(),
                    source.distribution(),
                    maps,
                    child_key,
                ) {
                    pairs.push((parent_key, source.data[offset].clone()));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }
}
