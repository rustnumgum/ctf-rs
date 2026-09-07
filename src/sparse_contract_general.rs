// Adapted from cc4s CTF contraction/{spctr_comm,spctr_tsr}.cxx and
// contraction.cxx map_extra_indices at f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Explicit-grid sparse-A/dense-B/dense-C contraction.
use crate::{
    algebra::{Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

// Sparse panels have a variable entry count, so broadcast the count before the
// serialized local-key/value pairs, as in spctr_replicate.
fn broadcast_sparse<E: Wire>(context: &Context<'_>, pairs: &mut Vec<(usize, E)>) {
    let mut count = [u64::try_from(pairs.len()).unwrap()];
    context.broadcast(0, &mut count);
    let width = 8 + E::WIDTH;
    let length = usize::try_from(count[0]).unwrap();
    let mut bytes = Vec::with_capacity(length * width);
    if context.rank() == 0 {
        for (key, value) in pairs.iter() {
            u64::try_from(*key).unwrap().encode(&mut bytes);
            value.encode(&mut bytes);
        }
    } else {
        bytes.resize(length * width, 0);
    }
    context.inner.broadcast(0, &mut bytes);
    if context.rank() != 0 {
        *pairs = bytes
            .chunks_exact(width)
            .map(|pair| {
                (
                    usize::try_from(u64::decode(&pair[..8])).unwrap(),
                    E::decode(&pair[8..]),
                )
            })
            .collect();
    }
}

// Move each stored entry once, from its original canonical owner to the mapped
// canonical owner. Replication is performed only by the source fiber layer.
fn sparse_on_roots<A: Semiring>(
    source: &SparseTensor<'_, '_, A>,
    target: &Distribution,
) -> Vec<(usize, A::Element)>
where
    A::Element: Wire,
{
    let context = source.context();
    let rank = context.rank();
    let width = 8 + A::Element::WIDTH;
    let mut buckets = vec![Vec::new(); context.size()];
    for (key, value) in source.local_pairs() {
        if source.distribution().owner(key) != rank {
            continue;
        }
        u64::try_from(key)
            .unwrap()
            .encode(&mut buckets[target.owner(key)]);
        value.encode(&mut buckets[target.owner(key)]);
    }
    let mut pairs = Vec::new();
    for bytes in context.inner.exchange(&buckets) {
        for pair in bytes.chunks_exact(width) {
            let key = usize::try_from(u64::decode(&pair[..8])).unwrap();
            pairs.push((target.local_offset(rank, key), A::Element::decode(&pair[8..])));
        }
    }
    pairs.sort_by_key(|pair| pair.0);
    pairs
}

// Dense mapped blocks likewise originate only on canonical roots; missing-axis
// replication is deferred to the explicit broadcasts below.
fn dense_on_roots<A: Semiring>(
    source: &Tensor<'_, '_, A>,
    target: &Distribution,
) -> Vec<A::Element>
where
    A::Element: Wire,
{
    let rank = source.context().rank();
    let requests: Vec<_> = (0..target.local_len())
        .filter_map(|offset| {
            target
                .global_key(rank, offset)
                .filter(|&key| target.owner(key) == rank)
                .map(|key| (offset, key))
        })
        .collect();
    let keys: Vec<_> = requests.iter().map(|&(_, key)| key).collect();
    let read = source.read(&keys);
    let mut values = vec![source.algebra().zero(); target.local_len()];
    for ((offset, _), value) in requests.into_iter().zip(read) {
        values[offset] = value;
    }
    values
}

macro_rules! define_mapped_contraction {
    ($(#[$attribute:meta])* $name:ident, $kernel:path
        $(, $function:ident : $function_type:ty)?) => {
    $(#[$attribute])*
    pub fn $name(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &Tensor<'_, '_, A>,
        indices_b: &str,
        topology: Topology,
        physical_labels: &str,
        virtual_factors: &[(u8, usize)],
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
        $( $function: $function_type, )?
    ) {
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        assert_eq!(topology.size(), self.context().size());
        assert!(physical_labels.is_ascii());
        assert_eq!(physical_labels.len(), topology.dimensions.len());

        let operands = [
            (indices_a, a.distribution()),
            (indices_b, b.distribution()),
            (indices_c, self.distribution()),
        ];
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        for &(indices, distribution) in &operands {
            assert!(indices.is_ascii());
            assert_eq!(indices.len(), distribution.shape.len());
            for (axis, label) in indices.bytes().enumerate() {
                assert!(
                    !indices.as_bytes()[..axis].contains(&label),
                    "sparse contraction requires unique labels per operand"
                );
                let dimension = distribution.shape[axis];
                if let Some(union_axis) = labels.iter().position(|&old| old == label) {
                    assert_eq!(dimensions[union_axis], dimension);
                } else {
                    labels.push(label);
                    dimensions.push(dimension);
                }
            }
        }
        for label in indices_a.bytes() {
            assert!(
                indices_b.as_bytes().contains(&label)
                    || indices_c.as_bytes().contains(&label),
                "sparse contraction does not accept A-only labels"
            );
        }

        let mut maps = vec![Mapping::Unmapped; labels.len()];
        for (topology_axis, label) in physical_labels.bytes().enumerate() {
            let union_axis = labels
                .iter()
                .position(|&candidate| candidate == label)
                .expect("each physical topology axis must name a union label");
            let occurrences = operands
                .iter()
                .filter(|(indices, _)| indices.as_bytes().contains(&label))
                .count();
            assert!(
                occurrences >= 2,
                "one-operand-only labels cannot be physically mapped"
            );
            maps[union_axis].augment_physical(&topology, topology_axis);
        }

        let mut virtual_dimensions = vec![1; labels.len()];
        for &(label, factor) in virtual_factors {
            assert!(factor > 0);
            let axis = labels.iter().position(|&candidate| candidate == label).unwrap();
            assert_eq!(virtual_dimensions[axis], 1);
            virtual_dimensions[axis] = factor;
            let total_phase = maps[axis].phase() * factor;
            maps[axis].augment_virtual(total_phase);
        }
        let index_maps: [Vec<usize>; 3] = std::array::from_fn(|operand|
            operands[operand].0.bytes().map(|label|labels.iter().position(|&x|x==label).unwrap()).collect());
        let mapped: [Distribution; 3] = std::array::from_fn(|operand| {
            let (indices, original) = operands[operand];
            Distribution::new(
                original.shape.clone(),
                topology.clone(),
                indices
                    .bytes()
                    .map(|label| {
                        maps[labels
                            .iter()
                            .position(|&candidate| candidate == label)
                            .unwrap()]
                        .clone()
                    })
                    .collect(),
            )
        });

        let mut communicators: [Vec<Context<'_>>; 3] = std::array::from_fn(|_| Vec::new());
        for (axis, label) in physical_labels.bytes().enumerate() {
            for operand in 0..3 {
                if !operands[operand].0.as_bytes().contains(&label) {
                    communicators[operand].push(topology.fiber(self.context(), axis));
                }
            }
        }

        let rank = self.context().rank();
        let mut sparse_a = sparse_on_roots(a, &mapped[0]);
        for communicator in &communicators[0] {
            broadcast_sparse(communicator, &mut sparse_a);
        }

        let mut dense_b = dense_on_roots(b, &mapped[1]);
        for communicator in &communicators[1] {
            communicator.broadcast(0, &mut dense_b);
        }
        let mut dense_c = dense_on_roots(self, &mapped[2]);
        let output_root = communicators[2]
            .iter()
            .all(|communicator| communicator.rank() == 0);
        let child_beta = if output_root {
            beta
        } else {
            self.algebra().zero()
        };
        let shapes: [Vec<usize>; 3] = std::array::from_fn(|operand| mapped[operand].block_shape());
        let block_sizes: [usize; 3] = std::array::from_fn(|operand| shapes[operand].iter().product());
        let count_a: usize = index_maps[0].iter().map(|&axis|virtual_dimensions[axis]).product();
        let mut sparse_blocks = vec![Vec::new(); count_a];
        for (offset,value) in sparse_a {
            sparse_blocks[offset / block_sizes[0]].push((offset % block_sizes[0],value));
        }
        let one = self.algebra().one();
        crate::sparse_virtual::execute(&virtual_dimensions,
            [&index_maps[0],&index_maps[1],&index_maps[2]],&child_beta,&one,|blocks,leaf_beta| {
        ($kernel)(
            self.algebra(),
            &shapes[0],
            indices_a,
            &sparse_blocks[blocks[0]],
            &shapes[1],
            indices_b,
            &dense_b[blocks[1]*block_sizes[1]..(blocks[1]+1)*block_sizes[1]],
            &shapes[2],
            indices_c,
            &mut dense_c[blocks[2]*block_sizes[2]..(blocks[2]+1)*block_sizes[2]],
            &alpha,
            leaf_beta,
            $( &$function, )?
        );
        });
        for communicator in &communicators[2] {
            communicator.reduce_monoid(self.algebra(), &mut dense_c, commutative, 0);
        }

        let contributions: Vec<_> = dense_c
            .into_iter()
            .enumerate()
            .filter_map(|(offset, value)| {
                mapped[2]
                    .global_key(rank, offset)
                    .filter(|&key| mapped[2].owner(key) == rank)
                    .map(|key| (key, value))
            })
            .collect();
        for group in communicators {
            for communicator in group {
                communicator.close();
            }
        }

        let zero = self.algebra().zero();
        self.transform(|_, value| *value = zero.clone());
        self.write_add(&contributions);
    }
    };
}

impl<A: Semiring + Clone> Tensor<'_, '_, A>
where
    A::Element: Wire,
{
    define_mapped_contraction!(
        /// Contract sparse `A` and dense `B` into dense `self` on an explicit
        /// label-to-topology-axis mapping. Labels are unique within each operand.
        /// A-only labels are unsupported by the source local sparse recursion;
        /// labels occurring in only one operand must not be physically mapped.
        /// Unmapped B-only and C-only labels are traversed locally.
        contract_from_sparse_dense_on,
        crate::sparse_sequential::sequential
    );

    define_mapped_contraction!(
        /// Apply a custom bivariate function to each stored sparse-A/dense-B
        /// pair on an explicit label-to-topology-axis mapping. Missing sparse A
        /// keys are not evaluated; stored zeros and dense B zeros are evaluated.
        /// The pinned general sparse kernel permits custom evaluation only for
        /// scalar A with a non-scalar union and unit alpha. Its all-scalar branch
        /// retains ordinary multiplication rather than invoking the function.
        contract_from_sparse_dense_function_on,
        crate::sparse_sequential::sequential_function,
        function: impl Fn(&A::Element, &A::Element) -> A::Element
    );
}
