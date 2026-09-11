//! Direct custom accumulation from compressed sparse symmetric storage.

use crate::{
    algebra::{Group, Monoid, Wire},
    sparse_symmetric::SparseSymmetricTensor,
    sparse_symmetric_search::Selected,
    tensor::Tensor,
};

impl<'c, 'r, OA> Tensor<'c, 'r, OA>
where
    OA: Monoid + Clone,
    OA::Element: Wire,
{
    /// Execute the selected compressed sparse mapping and reduce every input
    /// orbit onto the output labels with a caller-supplied in-place function.
    pub fn accumulate_sparse_symmetric_from_selected<IA: Group + Clone>(
        &mut self,
        output_indices: &str,
        input: &SparseSymmetricTensor<'c, 'r, IA>,
        input_indices: &str,
        selected: &Selected,
        mut function: impl FnMut(&IA::Element, &mut OA::Element),
    ) where IA::Element: Wire {
        assert!(std::ptr::eq(self.context(), input.context()));
        assert_eq!(selected.input.links(), input.distribution().links());
        assert_eq!(
            selected.input.distribution().shape,
            input.distribution().distribution().shape
        );
        assert_eq!(selected.output.shape, self.distribution().shape);
        let projection = Projection::new(
            &selected.input.distribution().shape,
            input_indices,
            &selected.output.shape,
            output_indices,
        );

        let mut mapped_input = (*input).clone();
        mapped_input.redistribute(selected.input.clone());
        let old_output = self.distribution().clone();
        let mut mapped_output = self.clone();
        mapped_output.redistribute(selected.output.clone());

        let mut buckets = vec![Vec::new(); self.context().size()];
        for (key, value) in mapped_input.local_orbit_pairs() {
            let coordinates = selected.input.distribution().decode_key(key);
            let Some(key) = projection.output_key(&coordinates, &selected.output) else {
                continue;
            };
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if selected.output.owns(rank, key) {
                    (key as u64).encode(bucket);
                    value.encode(bucket);
                }
            }
        }

        for bytes in self.context().inner.exchange(&buckets) {
            for contribution in bytes.chunks_exact(8 + IA::Element::WIDTH) {
                let key = u64::decode(&contribution[..8]) as usize;
                let value = IA::Element::decode(&contribution[8..]);
                let offset = selected.output.local_offset(self.context().rank(), key);
                function(&value, &mut mapped_output.data[offset]);
            }
        }

        mapped_output.redistribute(old_output);
        *self = mapped_output;
    }
}

struct Projection {
    input_axes: Vec<usize>,
    output_sources: Vec<usize>,
}

impl Projection {
    fn new(
        input_shape: &[usize],
        input_indices: &str,
        output_shape: &[usize],
        output_indices: &str,
    ) -> Self {
        assert!(input_indices.is_ascii() && output_indices.is_ascii());
        assert_eq!(input_shape.len(), input_indices.len());
        assert_eq!(output_shape.len(), output_indices.len());
        let input_labels = input_indices.as_bytes();
        let mut input_axes = Vec::with_capacity(input_labels.len());
        for (axis, &label) in input_labels.iter().enumerate() {
            let first = input_labels[..axis]
                .iter()
                .position(|&candidate| candidate == label)
                .unwrap_or(axis);
            assert_eq!(input_shape[first], input_shape[axis]);
            input_axes.push(first);
        }
        let output_sources = output_indices
            .bytes()
            .enumerate()
            .map(|(axis, label)| {
                let source = input_labels
                    .iter()
                    .position(|&candidate| candidate == label)
                    .expect("custom transform output label must occur in the sparse input");
                assert_eq!(output_shape[axis], input_shape[source]);
                source
            })
            .collect();
        Self {
            input_axes,
            output_sources,
        }
    }

    fn output_key(
        &self,
        input: &[usize],
        output: &crate::mapping::Distribution,
    ) -> Option<usize> {
        if self
            .input_axes
            .iter()
            .enumerate()
            .any(|(axis, &first)| input[axis] != input[first])
        {
            return None;
        }
        Some(output.encode_key(
            &self
                .output_sources
                .iter()
                .map(|&source| input[source])
                .collect::<Vec<_>>(),
        ))
    }
}
