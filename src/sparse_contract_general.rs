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
fn broadcast_sparse<E: Wire>(context: &Context<'_>, blocks: &mut [Vec<(usize, E)>]) {
    let mut count: Vec<_> = blocks.iter().map(|block| u64::try_from(block.len()).unwrap()).collect();
    context.broadcast(0, &mut count);
    let width = 8 + E::WIDTH;
    let length: usize = count.iter().map(|&n| usize::try_from(n).unwrap()).sum();
    let mut bytes = Vec::with_capacity(length * width);
    if context.rank() == 0 {
        for (key, value) in blocks.iter().flatten() {
            u64::try_from(*key).unwrap().encode(&mut bytes);
            value.encode(&mut bytes);
        }
    } else {
        bytes.resize(length * width, 0);
    }
    context.inner.broadcast(0, &mut bytes);
    if context.rank() != 0 {
        let mut pairs = bytes.chunks_exact(width);
        for (block, count) in blocks.iter_mut().zip(count) {
            *block = pairs.by_ref().take(usize::try_from(count).unwrap()).map(|pair| {
                (
                    usize::try_from(u64::decode(&pair[..8])).unwrap(),
                    E::decode(&pair[8..]),
                )
            })
            .collect();
        }
    }
}

// Move each stored entry once, from its original canonical owner to the mapped
// canonical owner. Replication is performed only by the source fiber layer.
fn sparse_on_roots<A: Semiring>(
    source: &SparseTensor<'_, '_, A>,
    target: &Distribution,
) -> Vec<Vec<(usize, A::Element)>>
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
    let phases: Vec<_> = target.mappings.iter().map(Mapping::phase).collect();
    let virtual_dimensions: Vec<_> = target.mappings.iter()
        .map(|mapping| mapping.phase() / mapping.physical_phase()).collect();
    let block_size: usize = target.block_shape().iter().product();
    let mut blocks = vec![Vec::new(); virtual_dimensions.iter().product()];
    for bytes in context.inner.exchange(&buckets) {
        for pair in bytes.chunks_exact(width) {
            let key = usize::try_from(u64::decode(&pair[..8])).unwrap();
            let block = target.local_offset(rank, key) / block_size;
            blocks[block].push((key, A::Element::decode(&pair[8..])));
        }
    }
    for block in &mut blocks { block.sort_by_key(|pair| pair.0); }
    let coordinates = target.topology.coordinates(rank);
    let metadata = crate::sparse_keys::KeyMetadata {
        shape: target.shape.clone(), phases, virtual_dimensions,
        physical_ranks: target.mappings.iter().map(|mapping| mapping.physical_rank(&coordinates)).collect(),
    };
    crate::sparse_keys::pin_blocks(&metadata, &blocks)
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

struct LabelMetadata {
    labels: Vec<u8>,
    dimensions: Vec<usize>,
    index_maps: [Vec<usize>; 3],
    occurrences: Vec<usize>,
}

impl LabelMetadata {
    fn new(distributions: [&Distribution; 3], indices: [&str; 3]) -> Self {
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        let index_maps: [Vec<usize>; 3] = std::array::from_fn(|operand| {
            assert!(indices[operand].is_ascii());
            assert_eq!(indices[operand].len(), distributions[operand].shape.len());
            indices[operand].bytes().enumerate().map(|(axis, label)| {
                assert!(
                    !indices[operand].as_bytes()[..axis].contains(&label),
                    "sparse contraction requires unique labels per operand"
                );
                if let Some(union_axis) = labels.iter().position(|&old| old == label) {
                    assert_eq!(dimensions[union_axis], distributions[operand].shape[axis]);
                    union_axis
                } else {
                    labels.push(label);
                    dimensions.push(distributions[operand].shape[axis]);
                    labels.len() - 1
                }
            }).collect()
        });
        for &label in &index_maps[0] {
            assert!(
                index_maps[1].contains(&label) || index_maps[2].contains(&label),
                "sparse contraction does not accept A-only labels"
            );
        }
        let occurrences = (0..labels.len())
            .map(|label| index_maps.iter().filter(|indices| indices.contains(&label)).count())
            .collect();
        Self { labels, dimensions, index_maps, occurrences }
    }
}

fn mapping_uses_axis(mapping: &Mapping, axis: usize) -> bool {
    match mapping {
        Mapping::Unmapped => false,
        Mapping::Physical { axis: mapped, child, .. } => {
            *mapped == axis || mapping_uses_axis(child, axis)
        }
        Mapping::Virtual { child, .. } => mapping_uses_axis(child, axis),
    }
}

fn execute_mapped<A, F>(
    output: &mut Tensor<'_, '_, A>,
    a: &SparseTensor<'_, '_, A>,
    b: &Tensor<'_, '_, A>,
    mapped: &[Distribution; 3],
    metadata: &LabelMetadata,
    beta: A::Element,
    commutative: bool,
    mut leaf: F,
) where
    A: Semiring + Clone,
    A::Element: Wire,
    F: FnMut(
        &A,
        &[usize],
        &[(usize, A::Element)],
        &[usize],
        &[A::Element],
        &[usize],
        &mut [A::Element],
        &A::Element,
    ),
{
    assert!(mapped.iter().all(|distribution| distribution.topology == mapped[0].topology));
    assert_eq!(mapped[0].topology.size(), output.context().size());
    for operand in 0..3 {
        assert_eq!(mapped[operand].shape.len(), metadata.index_maps[operand].len());
        for (axis, &label) in metadata.index_maps[operand].iter().enumerate() {
            assert_eq!(mapped[operand].shape[axis], metadata.dimensions[label]);
        }
    }

    let mut label_maps: Vec<Option<&Mapping>> = vec![None; metadata.labels.len()];
    for operand in 0..3 {
        for (axis, &label) in metadata.index_maps[operand].iter().enumerate() {
            if let Some(existing) = label_maps[label] {
                assert_eq!(existing, &mapped[operand].mappings[axis]);
            } else {
                label_maps[label] = Some(&mapped[operand].mappings[axis]);
            }
        }
    }
    let virtual_dimensions: Vec<_> = label_maps.iter().map(|mapping| {
        let mapping = mapping.unwrap();
        mapping.phase() / mapping.physical_phase()
    }).collect();

    let mut communicators: [Vec<Context<'_>>; 3] = std::array::from_fn(|_| Vec::new());
    for topology_axis in 0..mapped[0].topology.dimensions.len() {
        let mut axis_label = None;
        for operand in 0..3 {
            for (axis, &label) in metadata.index_maps[operand].iter().enumerate() {
                if mapping_uses_axis(&mapped[operand].mappings[axis], topology_axis) {
                    if let Some(existing) = axis_label {
                        assert_eq!(existing, label);
                    } else {
                        axis_label = Some(label);
                    }
                }
            }
        }
        let label = axis_label.expect("each topology axis must map a union label");
        assert!(
            metadata.occurrences[label] >= 2,
            "one-operand-only labels cannot be physically mapped"
        );
        for operand in 0..3 {
            let contains = metadata.index_maps[operand].contains(&label);
            let uses = mapped[operand].mappings.iter()
                .any(|mapping| mapping_uses_axis(mapping, topology_axis));
            assert_eq!(uses, contains, "shared labels must have aligned mappings");
            if !contains {
                communicators[operand].push(mapped[0].topology.fiber(output.context(), topology_axis));
            }
        }
    }

    let rank = output.context().rank();
    let mut sparse_blocks = sparse_on_roots(a, &mapped[0]);
    for communicator in &communicators[0] {
        broadcast_sparse(communicator, &mut sparse_blocks);
    }
    let mut dense_b = dense_on_roots(b, &mapped[1]);
    for communicator in &communicators[1] {
        communicator.broadcast(0, &mut dense_b);
    }
    let mut dense_c = dense_on_roots(output, &mapped[2]);
    let output_root = communicators[2].iter().all(|communicator| communicator.rank() == 0);
    let child_beta = if output_root { beta } else { output.algebra().zero() };
    let shapes: [Vec<usize>; 3] = std::array::from_fn(|operand| mapped[operand].block_shape());
    let block_sizes: [usize; 3] = std::array::from_fn(|operand| shapes[operand].iter().product());
    let one = output.algebra().one();
    crate::sparse_virtual::execute(
        &virtual_dimensions,
        [&metadata.index_maps[0], &metadata.index_maps[1], &metadata.index_maps[2]],
        &child_beta,
        &one,
        |blocks, leaf_beta| {
            leaf(
                output.algebra(),
                &shapes[0],
                &sparse_blocks[blocks[0]],
                &shapes[1],
                &dense_b[blocks[1] * block_sizes[1]..(blocks[1] + 1) * block_sizes[1]],
                &shapes[2],
                &mut dense_c[blocks[2] * block_sizes[2]..(blocks[2] + 1) * block_sizes[2]],
                leaf_beta,
            );
        },
    );
    for communicator in &communicators[2] {
        communicator.reduce_monoid(output.algebra(), &mut dense_c, commutative, 0);
    }
    let contributions: Vec<_> = dense_c.into_iter().enumerate().filter_map(|(offset, value)| {
        mapped[2].global_key(rank, offset)
            .filter(|&key| mapped[2].owner(key) == rank)
            .map(|key| (key, value))
    }).collect();
    for group in communicators {
        for communicator in group {
            communicator.close();
        }
    }
    let zero = output.algebra().zero();
    output.transform(|_, value| *value = zero.clone());
    output.write_add(&contributions);
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
        let distributions = [a.distribution(), b.distribution(), self.distribution()];
        let metadata = LabelMetadata::new(distributions, [indices_a, indices_b, indices_c]);
        let mut maps = vec![Mapping::Unmapped; metadata.labels.len()];
        for (topology_axis, label) in physical_labels.bytes().enumerate() {
            let union_axis = metadata.labels.iter().position(|&candidate| candidate == label)
                .expect("each physical topology axis must name a union label");
            assert!(metadata.occurrences[union_axis] >= 2,
                "one-operand-only labels cannot be physically mapped");
            maps[union_axis].augment_physical(&topology, topology_axis);
        }
        let mut assigned_virtual = vec![false; metadata.labels.len()];
        for &(label, factor) in virtual_factors {
            assert!(factor > 0);
            let axis = metadata.labels.iter().position(|&candidate| candidate == label).unwrap();
            assert!(!assigned_virtual[axis]);
            assigned_virtual[axis] = true;
            let total_phase = maps[axis].phase() * factor;
            maps[axis].augment_virtual(total_phase);
        }
        let mapped: [Distribution; 3] = std::array::from_fn(|operand| Distribution::new(
            distributions[operand].shape.clone(),
            topology.clone(),
            metadata.index_maps[operand].iter().map(|&label| maps[label].clone()).collect(),
        ));
        execute_mapped(self, a, b, &mapped, &metadata, beta, commutative,
            |algebra, shape_a, sparse_a, shape_b, dense_b, shape_c, dense_c, leaf_beta| {
                ($kernel)(
                    algebra, shape_a, indices_a, sparse_a, shape_b, indices_b, dense_b,
                    shape_c, indices_c, dense_c, &alpha, leaf_beta,
                    $( &$function, )?
                );
            });
    }
    };
}

impl<A: Semiring + Clone> Tensor<'_, '_, A>
where
    A::Element: Wire,
{
    /// Execute sparse-A/dense-B contraction with a reusable aligned grid plan.
    /// The plan contains mappings only; current tensor values, coefficients and
    /// reduction commutativity are supplied for every execution.
    pub fn contract_sparse_with_plan(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        plan: &crate::planning::GridPlan,
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) {
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        assert_eq!(plan.signature().topology().size(), self.context().size());
        assert!(plan.matches(
            [a.distribution(), b.distribution(), self.distribution()],
            [indices_a, indices_b, indices_c],
        ));
        let metadata = LabelMetadata::new(
            [a.distribution(), b.distribution(), self.distribution()],
            [indices_a, indices_b, indices_c],
        );
        assert_eq!(&metadata.index_maps, plan.signature().indices());
        execute_mapped(
            self,
            a,
            b,
            plan.mapped_distributions(),
            &metadata,
            beta,
            commutative,
            |algebra, shape_a, sparse_a, shape_b, dense_b, shape_c, dense_c, leaf_beta| {
                crate::sparse_sequential::sequential(
                    algebra,
                    shape_a,
                    indices_a,
                    sparse_a,
                    shape_b,
                    indices_b,
                    dense_b,
                    shape_c,
                    indices_c,
                    dense_c,
                    &alpha,
                    leaf_beta,
                );
            },
        );
    }

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
