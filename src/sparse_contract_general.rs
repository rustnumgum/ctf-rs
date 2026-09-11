// Adapted from cc4s CTF contraction/{spctr_comm,spctr_tsr}.cxx and
// contraction.cxx map_extra_indices at f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Explicit-grid sparse-A/dense-B/dense-C contraction.
use crate::{
    algebra::{Arithmetic, Monoid, Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    sparse_formats::{Coo, Csr},
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
fn sparse_on_roots<A: Monoid>(
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

struct WeighIndex {
    axis_a: usize,
    axis_b: usize,
    expand_a: bool,
    fresh_label: u8,
}

fn canonical_nnz<A: Monoid>(tensor: &SparseTensor<'_, '_, A>) -> u64
where
    A::Element: Wire,
{
    let rank = tensor.context().rank();
    let local = tensor.local_pairs().into_iter()
        .filter(|(key, _)| tensor.distribution().owner(*key) == rank)
        .count() as u64;
    tensor.context().all_reduce(&Arithmetic::<u64>::new(), &local)
}

fn sparse_weigh_index(
    distributions: [&Distribution; 3],
    indices: [&str; 3],
    nonzeros: [Option<u64>; 2],
) -> Option<WeighIndex> {
    let metadata = LabelMetadata::new(distributions, indices);
    let mut size_a_1 = 1u64;
    let mut size_a_2 = 1u64;
    let mut size_b_1 = 1u64;
    let mut size_b_2 = 1u64;
    let mut last = None;
    for label in 0..metadata.labels.len() {
        let positions = std::array::from_fn::<_, 3, _>(|operand| {
            metadata.index_maps[operand].iter().position(|&candidate| candidate == label)
        });
        match positions {
            [Some(axis_a), Some(axis_b), Some(_)] => {
                let extent_a = distributions[0].shape[axis_a] as u64;
                let extent_b = distributions[1].shape[axis_b] as u64;
                size_a_1 *= extent_a;
                size_a_2 *= extent_a;
                size_b_1 *= extent_b;
                size_b_2 *= extent_b;
                last = Some((axis_a, axis_b));
            }
            [Some(axis_a), Some(axis_b), None] => {
                size_a_1 *= distributions[0].shape[axis_a] as u64;
                size_b_1 *= distributions[1].shape[axis_b] as u64;
            }
            [Some(axis_a), None, Some(_)] => {
                size_a_1 *= distributions[0].shape[axis_a] as u64;
            }
            [None, Some(axis_b), Some(_)] => {
                size_b_1 *= distributions[1].shape[axis_b] as u64;
            }
            _ => {}
        }
    }
    let (axis_a, axis_b) = last?;
    let size_a = nonzeros[0].unwrap_or(distributions[0].global_len() as u64)
        .max(size_a_1.min(size_a_2));
    let size_b = nonzeros[1].unwrap_or(distributions[1].global_len() as u64)
        .max(size_b_1.min(size_b_2));
    let fresh_label = (b'a'..=b'z')
        .chain(b'A'..=b'Z')
        .chain(b'0'..=b'9')
        .chain(1..=127)
        .find(|label| !metadata.labels.contains(label))
        .expect("sparse contraction exhausted ASCII labels");
    Some(WeighIndex {
        axis_a,
        axis_b,
        expand_a: nonzeros[0].is_some() && (nonzeros[1].is_none() || size_a < size_b),
        fresh_label,
    })
}

fn insert_label(indices: &str, axis: usize, label: u8) -> String {
    let mut indices = indices.as_bytes().to_vec();
    indices.insert(axis, label);
    String::from_utf8(indices).unwrap()
}

fn replace_label(indices: &str, axis: usize, label: u8) -> String {
    let mut indices = indices.as_bytes().to_vec();
    indices[axis] = label;
    String::from_utf8(indices).unwrap()
}

fn expand_sparse_diagonal<'c, 'r, A>(
    tensor: &SparseTensor<'c, 'r, A>,
    axis: usize,
) -> SparseTensor<'c, 'r, A>
where
    A: Monoid + Clone,
    A::Element: Wire,
{
    let mut shape = tensor.distribution().shape.clone();
    shape.insert(axis, shape[axis]);
    let distribution = Distribution::cyclic(shape, tensor.context().size());
    let rank = tensor.context().rank();
    let pairs: Vec<_> = tensor.local_pairs().into_iter().filter_map(|(key, value)| {
        if tensor.distribution().owner(key) != rank {
            return None;
        }
        let mut coordinates = tensor.distribution().decode_key(key);
        let diagonal = coordinates[axis];
        coordinates.insert(axis, diagonal);
        Some((distribution.encode_key(&coordinates), value))
    }).collect();
    let mut expanded = SparseTensor::new(tensor.context(), distribution, tensor.algebra().clone());
    expanded.write_add(&pairs);
    expanded
}

/// Hadamard-index elimination step (commit d5861de), A side: expand the
/// diagonal `weigh` selected on `a` and relabel the matching axis on B's
/// index string. Shared by every `contract_sparse*` recursion below.
fn hadamard_expand_a<'c, 'r, A>(
    a: &SparseTensor<'c, 'r, A>,
    indices_a: &str,
    indices_b: &str,
    weigh: &WeighIndex,
) -> (SparseTensor<'c, 'r, A>, String, String)
where
    A: Monoid + Clone,
    A::Element: Wire,
{
    let expanded = expand_sparse_diagonal(a, weigh.axis_a);
    let expanded_indices = insert_label(indices_a, weigh.axis_a, weigh.fresh_label);
    let relabeled_b = replace_label(indices_b, weigh.axis_b, weigh.fresh_label);
    (expanded, expanded_indices, relabeled_b)
}

/// Hadamard-index elimination step, B side: the mirror of
/// [`hadamard_expand_a`], expanding `b` and relabeling A's index string.
fn hadamard_expand_b<'c, 'r, B>(
    indices_a: &str,
    b: &SparseTensor<'c, 'r, B>,
    indices_b: &str,
    weigh: &WeighIndex,
) -> (String, SparseTensor<'c, 'r, B>, String)
where
    B: Monoid + Clone,
    B::Element: Wire,
{
    let expanded = expand_sparse_diagonal(b, weigh.axis_b);
    let relabeled_a = replace_label(indices_a, weigh.axis_a, weigh.fresh_label);
    let expanded_indices = insert_label(indices_b, weigh.axis_b, weigh.fresh_label);
    (relabeled_a, expanded, expanded_indices)
}

/// Hadamard-index elimination for two sparse operands (source contraction.cxx
/// map_extra_indices): expand the diagonal on whichever side `weigh`
/// selected, relabel the other side's matching axis, and recurse the
/// contraction once through `recurse` with the (possibly expanded) operands
/// and their (possibly relabeled) index strings. `contract_sparse` (both
/// operands the same algebra) and `contract_sparse_function` (heterogeneous
/// operands) both drive this with their own `recurse` closure.
fn eliminate_hadamard_index<'c, 'r, A, B, T>(
    a: &SparseTensor<'c, 'r, A>,
    indices_a: &str,
    b: &SparseTensor<'c, 'r, B>,
    indices_b: &str,
    weigh: WeighIndex,
    recurse: impl FnOnce(&SparseTensor<'c, 'r, A>, &str, &SparseTensor<'c, 'r, B>, &str) -> T,
) -> T
where
    A: Monoid + Clone,
    A::Element: Wire,
    B: Monoid + Clone,
    B::Element: Wire,
{
    if weigh.expand_a {
        let (expanded, expanded_indices, relabeled_b) = hadamard_expand_a(a, indices_a, indices_b, &weigh);
        recurse(&expanded, &expanded_indices, b, &relabeled_b)
    } else {
        let (relabeled_a, expanded, expanded_indices) = hadamard_expand_b(indices_a, b, indices_b, &weigh);
        recurse(a, &relabeled_a, &expanded, &expanded_indices)
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

#[allow(clippy::too_many_arguments)]
fn execute_raw_panels<A: Semiring>(
    communicators: &[[Option<Context<'_>>; 3]],
    execution: &crate::sparse_mapped_cost::Execution,
    level: usize,
    layers: crate::sparse_2d::Layers,
    algebra: &A,
    indices: [&str; 3],
    alpha: &A::Element,
    a: &[Vec<(usize, A::Element)>],
    b: &[Vec<A::Element>],
    mut c: Vec<Vec<A::Element>>,
    beta: A::Element,
) -> Vec<Vec<A::Element>>
where
    A::Element: Wire,
{
    if level == execution.panels.len() {
        let one = algebra.one();
        let sizes: [usize; 3] = execution.block_shapes.each_ref()
            .map(|shape| shape.iter().product());
        crate::sparse_virtual::execute(
            &execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice),
            &beta,
            &one,
            |blocks, leaf_beta| {
                crate::sparse_sequential::sequential(
                    algebra,
                    &execution.block_shapes[0],
                    indices[0],
                    &a[blocks[0]],
                    &execution.block_shapes[1],
                    indices[1],
                    &b[blocks[1]],
                    &execution.block_shapes[2],
                    indices[2],
                    &mut c[blocks[2]][..sizes[2]],
                    alpha,
                    leaf_beta,
                );
            },
        );
        return c;
    }

    let panel = &execution.panels[level];
    let plans: [crate::sparse_2d::Panel<'_, '_>; 3] = std::array::from_fn(|operand| crate::sparse_2d::Panel {
        comm: communicators[level][operand].as_ref(),
        outer: panel.operands[operand].outer,
        inner: panel.operands[operand].inner,
    });
    let result = crate::sparse_2d::execute_pairs_dense(
        algebra,
        panel.edge,
        layers,
        plans[0],
        plans[1],
        plans[2],
        a,
        b,
        c,
        beta,
        |a, b, c, child_beta, child_layers| {
            execute_raw_panels(
                communicators,
                execution,
                level + 1,
                child_layers,
                algebra,
                indices,
                alpha,
                a,
                b,
                c,
                child_beta,
            )
        },
    );
    result
}

fn dense_blocks<E: Clone>(values: Vec<E>, block_size: usize) -> Vec<Vec<E>> {
    assert!(block_size > 0 && values.len() % block_size == 0);
    values.chunks_exact(block_size).map(<[E]>::to_vec).collect()
}

fn local_key_metadata(distribution: &Distribution, rank: usize) -> crate::sparse_keys::KeyMetadata {
    let coordinates = distribution.topology.coordinates(rank);
    crate::sparse_keys::KeyMetadata {
        shape: distribution.shape.clone(),
        phases: distribution.mappings.iter().map(Mapping::phase).collect(),
        virtual_dimensions: distribution.mappings.iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase()).collect(),
        physical_ranks: distribution.mappings.iter()
            .map(|mapping| mapping.physical_rank(&coordinates)).collect(),
    }
}

fn folded_matricization(
    execution: &crate::sparse_mapped_cost::Execution,
    descriptor: &crate::partial_fold::Descriptor,
    operand: usize,
) -> crate::sparse_matricize::Matricization {
    let mut masks = vec![0u8; execution.virtual_dimensions.len()];
    for source in 0..3 {
        for &label in &execution.indices[source] { masks[label] |= 1 << source; }
    }
    let row_mask = [0b101, 0b011, 0b101][operand];
    let layout = &descriptor.layouts[operand];
    assert_eq!(layout.folded_shape.len(), layout.inner_ordering.len(),
        "selected sparse fold must matricize every local tensor group");
    // Source nrow_idx counts row labels in the tensor's original dimension
    // order. inner_ordering, rather than folded_indices, places those groups
    // at the matricized prefix selected by permutation 1 (A,C,B).
    let row_dimensions = layout.folded_indices.iter()
        .filter(|&&label| masks[label] == row_mask).count();
    assert!(layout.inner_ordering[..row_dimensions].iter()
        .all(|&group| masks[layout.folded_indices[group]] == row_mask));
    assert!(layout.inner_ordering[row_dimensions..].iter()
        .all(|&group| masks[layout.folded_indices[group]] != row_mask));
    let shape = execution.block_shapes[operand].clone();
    crate::sparse_matricize::Matricization {
        padded_shape: shape.clone(),
        links: vec![crate::symmetry::Symmetry::NS; shape.len()],
        folded_shape: layout.folded_shape.clone(),
        reverse_ordering: layout.inner_ordering.clone(),
        row_dimensions,
        phases: vec![1; shape.len()],
        shape,
    }
}

fn folded_dematricization(
    execution: &crate::sparse_mapped_cost::Execution,
    descriptor: &crate::partial_fold::Descriptor,
) -> crate::sparse_matricize::Dematricization {
    let matrix = folded_matricization(execution, descriptor, 2);
    crate::sparse_matricize::Dematricization {
        phase_ranks: vec![0; matrix.shape.len()],
        phases: matrix.phases,
        row_dimensions: matrix.row_dimensions,
        reverse_ordering: matrix.reverse_ordering,
        shape: matrix.shape,
    }
}

fn broadcast_coo<E: Wire + Clone>(context: &Context<'_>, root: usize,
    blocks: &mut Vec<Coo<E>>) {
    let width = 16 + E::WIDTH;
    let mut shapes = vec![0u64; 3 * blocks.len()];
    if context.rank() == root {
        for (shape, block) in shapes.chunks_exact_mut(3).zip(blocks.iter()) {
            let (rows, columns) = block.shape();
            shape.copy_from_slice(&[rows.try_into().unwrap(), columns.try_into().unwrap(),
                block.entries().len().try_into().unwrap()]);
        }
    }
    context.broadcast(root, &mut shapes);
    let entries: usize = shapes.chunks_exact(3)
        .map(|shape| usize::try_from(shape[2]).unwrap()).sum();
    let mut bytes = Vec::with_capacity(entries * width);
    if context.rank() == root {
        for (row, column, value) in blocks.iter().flat_map(Coo::entries) {
            u64::try_from(*row).unwrap().encode(&mut bytes);
            u64::try_from(*column).unwrap().encode(&mut bytes);
            value.encode(&mut bytes);
        }
    } else {
        bytes.resize(entries * width, 0);
    }
    context.inner.broadcast(root, &mut bytes);
    if context.rank() != root {
        let mut encoded = bytes.chunks_exact(width);
        *blocks = shapes.chunks_exact(3).map(|shape| {
            let entries = (0..shape[2]).map(|_| {
                let entry = encoded.next().unwrap();
                (usize::try_from(u64::decode(&entry[..8])).unwrap(),
                 usize::try_from(u64::decode(&entry[8..16])).unwrap(),
                 E::decode(&entry[16..]))
            }).collect();
            Coo::new(shape[0].try_into().unwrap(), shape[1].try_into().unwrap(), entries)
        }).collect();
    }
}

fn coo_batch<E: Clone>(matrix: &Coo<E>, rows: usize, columns: usize,
    batch: usize, batches: usize) -> Coo<E> {
    assert_eq!(matrix.shape(), (rows, columns * batches));
    let start = batch * columns;
    Coo::new(rows, columns, matrix.entries().iter().filter_map(|(row, column, value)| {
        let column = *column - 1;
        (column >= start && column < start + columns)
            .then(|| (*row, column - start + 1, value.clone()))
    }).collect())
}

fn join_batches<E: Clone>(matrices: &[Coo<E>], rows: usize, columns: usize) -> Coo<E> {
    let entries = matrices.iter().enumerate().flat_map(|(batch, matrix)| {
        assert_eq!(matrix.shape(), (rows, columns));
        matrix.entries().iter().map(move |(row, column, value)| {
            (*row, *column + batch * columns, value.clone())
        })
    }).collect();
    Coo::new(rows, columns * matrices.len(), entries)
}

fn panels<'c, 'r>(context: &'c Context<'r>, topology: &Topology,
    execution: &crate::sparse_mapped_cost::Execution) -> Vec<[Option<Context<'c>>; 3]> {
    execution.panels.iter().map(|panel| std::array::from_fn(|operand| {
        panel.operands[operand].topology_axis.map(|axis| topology.fiber(context, axis))
    })).collect()
}

fn panel_plans<'a, 'r>(contexts: &'a [[Option<Context<'r>>; 3]],
    execution: &'a crate::sparse_mapped_cost::Execution, level: usize)
    -> [crate::sparse_2d::Panel<'a, 'r>; 3] {
    let panel = &execution.panels[level];
    std::array::from_fn(|operand| crate::sparse_2d::Panel {
        comm: contexts[level][operand].as_ref(),
        outer: panel.operands[operand].outer,
        inner: panel.operands[operand].inner,
    })
}

fn close_contexts(contexts: Vec<[Option<Context<'_>>; 3]>) {
    for level in contexts {
        for context in level.into_iter().flatten() { context.close(); }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_coo_dense_panels<A: Semiring>(
    contexts: &[[Option<Context<'_>>; 3]], execution: &crate::sparse_mapped_cost::Execution,
    level: usize, layers: crate::sparse_2d::Layers, descriptor: &crate::partial_fold::Descriptor,
    algebra: &A, alpha: &A::Element, a: &[Coo<A::Element>], b: &[Vec<A::Element>],
    c: Vec<Vec<A::Element>>, beta: A::Element,
) -> Vec<Vec<A::Element>> where A::Element: Wire {
    if level == execution.panels.len() {
        let mut c = c;
        let one = algebra.one();
        crate::sparse_virtual::execute(&execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice), &beta, &one,
            |blocks, leaf_beta| {
            for batch in 0..descriptor.batches {
                let a = coo_batch(&a[blocks[0]], descriptor.m, descriptor.k,
                    batch, descriptor.batches);
                let b = &b[blocks[1]][batch * descriptor.k * descriptor.n
                    ..(batch + 1) * descriptor.k * descriptor.n];
                let c = &mut c[blocks[2]][batch * descriptor.m * descriptor.n
                    ..(batch + 1) * descriptor.m * descriptor.n];
                let zero = algebra.zero();
                if leaf_beta == &zero {
                    c.fill(zero);
                } else if leaf_beta != &one {
                    for value in c.iter_mut() { *value = algebra.multiply(leaf_beta, value); }
                }
                a.coomm(algebra, descriptor.n, b, alpha, &one, c);
            }
        });
        return c;
    }
    let plan = panel_plans(contexts, execution, level);
    crate::sparse_2d::execute_coo_dense(algebra, execution.panels[level].edge, layers,
        plan[0], plan[1], plan[2], a, b, c, beta,
        |a, b, c, beta, layers| execute_coo_dense_panels(contexts, execution, level + 1,
            layers, descriptor, algebra, alpha, a, b, c, beta))
}

#[allow(clippy::too_many_arguments)]
fn execute_csr_dense_panels<A: Semiring>(
    contexts: &[[Option<Context<'_>>; 3]], execution: &crate::sparse_mapped_cost::Execution,
    level: usize, layers: crate::sparse_2d::Layers, descriptor: &crate::partial_fold::Descriptor,
    algebra: &A, alpha: &A::Element, a: &[Csr<A::Element>], b: &[Vec<A::Element>],
    c: Vec<Vec<A::Element>>, beta: A::Element,
) -> Vec<Vec<A::Element>> where A::Element: Wire {
    if level == execution.panels.len() {
        let mut c = c;
        let one = algebra.one();
        crate::sparse_virtual::execute(&execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice), &beta, &one,
            |blocks, leaf_beta| {
            for batch in 0..descriptor.batches {
                let a = coo_batch(&a[blocks[0]].to_coo(), descriptor.m, descriptor.k,
                    batch, descriptor.batches).to_csr();
                let b = &b[blocks[1]][batch * descriptor.k * descriptor.n
                    ..(batch + 1) * descriptor.k * descriptor.n];
                let c = &mut c[blocks[2]][batch * descriptor.m * descriptor.n
                    ..(batch + 1) * descriptor.m * descriptor.n];
                a.multiply_dense(descriptor.n, b, alpha, leaf_beta, c, algebra);
            }
        });
        return c;
    }
    let plan = panel_plans(contexts, execution, level);
    crate::sparse_2d::execute_csr_dense(algebra, execution.panels[level].edge, layers,
        plan[0], plan[1], plan[2], a, b, c, beta,
        |a, b, c, beta, layers| execute_csr_dense_panels(contexts, execution, level + 1,
            layers, descriptor, algebra, alpha, a, b, c, beta))
}

#[allow(clippy::too_many_arguments)]
fn execute_csr_sparse_dense_panels<A: Semiring>(
    contexts: &[[Option<Context<'_>>; 3]], execution: &crate::sparse_mapped_cost::Execution,
    level: usize, layers: crate::sparse_2d::Layers, descriptor: &crate::partial_fold::Descriptor,
    algebra: &A, alpha: &A::Element, a: &[Csr<A::Element>], b: &[Csr<A::Element>],
    c: Vec<Vec<A::Element>>, beta: A::Element,
) -> Vec<Vec<A::Element>> where A::Element: Wire {
    if level == execution.panels.len() {
        let mut c = c;
        let one = algebra.one();
        crate::sparse_virtual::execute(&execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice), &beta, &one,
            |blocks, leaf_beta| {
            for batch in 0..descriptor.batches {
                let a = coo_batch(&a[blocks[0]].to_coo(), descriptor.m, descriptor.k,
                    batch, descriptor.batches).to_csr();
                let b = coo_batch(&b[blocks[1]].to_coo(), descriptor.k, descriptor.n,
                    batch, descriptor.batches).to_csr();
                let c = &mut c[blocks[2]][batch * descriptor.m * descriptor.n
                    ..(batch + 1) * descriptor.m * descriptor.n];
                super::gemm::kernel::csr_sparse_dense(&a, &b, c, alpha, leaf_beta, algebra);
            }
        });
        return c;
    }
    let plan = panel_plans(contexts, execution, level);
    crate::sparse_2d::execute_csr_sparse_dense(algebra, execution.panels[level].edge, layers,
        plan[0], plan[1], plan[2], a, b, c, beta,
        |a, b, c, beta, layers| execute_csr_sparse_dense_panels(contexts, execution, level + 1,
            layers, descriptor, algebra, alpha, a, b, c, beta))
}

#[allow(clippy::too_many_arguments)]
fn execute_csr_panels<A: Semiring>(
    contexts: &[[Option<Context<'_>>; 3]], execution: &crate::sparse_mapped_cost::Execution,
    level: usize, layers: crate::sparse_2d::Layers, descriptor: &crate::partial_fold::Descriptor,
    algebra: &A, alpha: &A::Element, a: &[Csr<A::Element>], b: &[Csr<A::Element>],
    c: Vec<Csr<A::Element>>, beta: A::Element,
) -> Vec<Csr<A::Element>> where A::Element: Wire {
    if level == execution.panels.len() {
        let mut c = c;
        let one = algebra.one();
        crate::sparse_virtual::execute(&execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice), &beta, &one,
            |blocks, leaf_beta| {
            let mut batches = Vec::with_capacity(descriptor.batches);
            for batch in 0..descriptor.batches {
                let a = coo_batch(&a[blocks[0]].to_coo(), descriptor.m, descriptor.k,
                    batch, descriptor.batches).to_csr();
                let b = coo_batch(&b[blocks[1]].to_coo(), descriptor.k, descriptor.n,
                    batch, descriptor.batches).to_csr();
                let old = coo_batch(&c[blocks[2]].to_coo(), descriptor.m, descriptor.n,
                    batch, descriptor.batches).to_csr();
                batches.push(a.multiply_sparse(&b, alpha, leaf_beta, Some(&old), algebra).to_coo());
            }
            c[blocks[2]] = join_batches(&batches, descriptor.m, descriptor.n).to_csr();
        });
        return c;
    }
    let plan = panel_plans(contexts, execution, level);
    crate::sparse_2d::execute_csr(algebra, execution.panels[level].edge, layers,
        plan[0], plan[1], plan[2], a, b, c, beta,
        |a, b, c, beta, layers| execute_csr_panels(contexts, execution, level + 1,
            layers, descriptor, algebra, alpha, a, b, c, beta))
}

fn selected_plan(
    selected: &crate::sparse_search::Selected,
    indices: [&str; 3],
    element_sizes: [usize; 3],
    custom: bool,
) -> crate::sparse_mapped_cost::Plan {
    let sparse = selected.pattern.sparse();
    let storage = std::array::from_fn(|operand| crate::sparse_cost::Storage {
        sparse: sparse[operand],
        element_size: element_sizes[operand],
        pair_size: 8 + element_sizes[operand],
        dense_virtual_size: if sparse[operand] {
            selected.distributions[operand].block_shape().iter().product()
        } else { 0 },
        custom_addition: false,
    });
    crate::sparse_mapped_cost::build(
        selected.distributions.each_ref(),
        indices,
        crate::sparse_mapped_cost::Inputs {
            storage,
            fractions: crate::sparse_cost::Fractions { a: 1., b: 1., c: 1. },
            custom,
        },
        selected.fold.as_ref(),
        selected.pattern.coo_kernel(),
    ).expect("selected sparse raw plan is structurally unsupported")
}

fn replicate_coo<E: Wire + Clone>(communicators: &[Context<'_>], blocks: &mut Vec<Coo<E>>) {
    for communicator in communicators { broadcast_coo(communicator, 0, blocks); }
}

fn reduce_csr<A: Semiring>(communicators: &[Context<'_>], algebra: &A,
    blocks: &mut Vec<Csr<A::Element>>) where A::Element: Wire {
    for communicator in communicators {
        for block in blocks.iter_mut() {
            let shape = block.shape();
            *block = block.reduce(communicator, 0, algebra).unwrap_or_else(|| {
                Coo::new(shape.0, shape.1, Vec::new()).to_csr()
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_custom_csr_panels<C, EA, EB, F, G>(
    contexts: &[[Option<Context<'_>>; 3]],
    execution: &crate::sparse_mapped_cost::Execution,
    level: usize,
    layers: crate::sparse_2d::Layers,
    descriptor: &crate::partial_fold::Descriptor,
    algebra: &C,
    a: &[Csr<EA>],
    b: &[Csr<EB>],
    mut c: Vec<Csr<C::Element>>,
    function: &F,
    accumulate: &G,
) -> Vec<Csr<C::Element>>
where
    C: Monoid,
    C::Element: Wire,
    EA: Wire + Clone,
    EB: Wire + Clone,
    F: Fn(&EA, &EB) -> C::Element,
    G: Fn(C::Element, &mut C::Element),
{
    if level == execution.panels.len() {
        crate::sparse_virtual::execute(
            &execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice),
            &false,
            &true,
            |blocks, _| {
                let mut batches = Vec::with_capacity(descriptor.batches);
                for batch in 0..descriptor.batches {
                    let aa = coo_batch(&a[blocks[0]].to_coo(), descriptor.m, descriptor.k,
                        batch, descriptor.batches).to_csr();
                    let bb = coo_batch(&b[blocks[1]].to_coo(), descriptor.k, descriptor.n,
                        batch, descriptor.batches).to_csr();
                    let old = coo_batch(&c[blocks[2]].to_coo(), descriptor.m, descriptor.n,
                        batch, descriptor.batches).to_csr();
                    batches.push(crate::kernel::csr_sparse(
                        &aa,
                        &bb,
                        Some(&old),
                        |left, right| function(left, right),
                        |value, output| accumulate(value, output),
                    ).to_coo());
                }
                c[blocks[2]] = join_batches(&batches, descriptor.m, descriptor.n).to_csr();
            },
        );
        return c;
    }

    let plans = panel_plans(contexts, execution, level);
    assert!(!(plans[0].comm.is_some() && plans[1].comm.is_some() && plans[2].comm.is_some()));
    let edge = execution.panels[level].edge;
    let (count, index, next) = crate::sparse_2d::schedule(edge, layers);
    let moving_output = plans[2].comm.is_some();
    let mut contributions: Vec<_> = if moving_output {
        c.iter().map(crate::sparse_2d::csr_empty).collect()
    } else {
        Vec::new()
    };
    for step in (index..edge).step_by(count) {
        let aa = crate::sparse_2d::csr_operand(plans[0], a, step, edge);
        let bb = crate::sparse_2d::csr_operand(plans[1], b, step, edge);
        if let Some(context) = plans[2].comm {
            let positions = plans[2].operand_positions(c.len(), step, edge);
            let work: Vec<_> = positions.iter()
                .map(|&position| crate::sparse_2d::csr_empty(&c[position])).collect();
            let work = execute_custom_csr_panels(contexts, execution, level + 1, next,
                descriptor, algebra, &aa, &bb, work, function, accumulate);
            let owner = step % context.size();
            for (position, block) in positions.into_iter().zip(work) {
                if let Some(reduced) = block.reduce(context, owner, algebra) {
                    contributions[position] = reduced;
                }
            }
        } else if plans[2].inner == 0 {
            c = execute_custom_csr_panels(contexts, execution, level + 1, next,
                descriptor, algebra, &aa, &bb, c, function, accumulate);
        } else {
            let positions = plans[2].operand_positions(c.len(), step, edge);
            let work: Vec<_> = if plans[2].outer == 1 {
                positions.iter().map(|&position| c[position].clone()).collect()
            } else {
                positions.iter().map(|&position| crate::sparse_2d::csr_empty(&c[position])).collect()
            };
            let work = execute_custom_csr_panels(contexts, execution, level + 1, next,
                descriptor, algebra, &aa, &bb, work, function, accumulate);
            for (position, block) in positions.into_iter().zip(work) { c[position] = block; }
        }
    }
    if moving_output {
        for (old, contribution) in c.iter_mut().zip(contributions) {
            *old = old.add(&contribution, algebra);
        }
    }
    c
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
    /// Eliminate sparse Hadamard indices before selecting and executing the
    /// sparse-A/dense-B/dense-C contraction.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_sparse(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        cache: &mut crate::sparse_search::SearchCache<'_, '_>,
        alpha: A::Element,
        beta: A::Element,
        output_fraction: Option<f64>,
        commutative: bool,
    ) -> Result<(), crate::sparse_search::Error> {
        let nonzeros_a = canonical_nnz(a);
        let distributions = [a.distribution(), b.distribution(), self.distribution()];
        let indices = [indices_a, indices_b, indices_c];
        if let Some(weigh) = sparse_weigh_index(distributions, indices, [Some(nonzeros_a), None]) {
            assert!(weigh.expand_a, "dense Hadamard-index expansion is unsupported");
            let (expanded, expanded_indices, relabeled_b) =
                hadamard_expand_a(a, indices_a, indices_b, &weigh);
            return self.contract_sparse(
                indices_c,
                &expanded,
                &expanded_indices,
                b,
                &relabeled_b,
                cache,
                alpha,
                beta,
                output_fraction,
                commutative,
            );
        }
        let selected = cache.prepare(
            distributions,
            indices,
            [Some(nonzeros_a), None, None],
            output_fraction,
        )?.expect("sparse contraction search found no eligible plan").clone();
        self.contract_sparse_from_selected(
            indices_c,
            a,
            indices_a,
            b,
            indices_b,
            &selected,
            alpha,
            beta,
            commutative,
        );
        Ok(())
    }

    /// Execute the source unfolded sparse-A/dense-B/dense-C tree for arbitrary
    /// preflight-valid raw mappings, including shared-label mapping mismatches
    /// that require nested 2D panel movement.
    pub fn contract_sparse_from_mapped(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        mapped: [Distribution; 3],
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) {
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, b.distribution().shape);
        assert_eq!(mapped[2].shape, self.distribution().shape);
        assert!(mapped.iter().all(|distribution| {
            distribution.topology == mapped[0].topology
                && distribution.topology.size() == self.context().size()
        }));
        let metadata = LabelMetadata::new(
            [a.distribution(), b.distribution(), self.distribution()],
            [indices_a, indices_b, indices_c],
        );
        let element_size = A::Element::WIDTH;
        let pair_size = 8 + element_size;
        let dense_virtual_size: usize = mapped[0].block_shape().iter().product();
        let storage = [
            crate::sparse_cost::Storage { sparse: true, element_size, pair_size,
                dense_virtual_size, custom_addition: false },
            crate::sparse_cost::Storage { sparse: false, element_size, pair_size,
                dense_virtual_size: 0, custom_addition: false },
            crate::sparse_cost::Storage { sparse: false, element_size, pair_size,
                dense_virtual_size: 0, custom_addition: false },
        ];
        // Reuse only structural execution descriptors here. Fractions do not
        // affect layout assembly; no cost estimate is evaluated by this call.
        let plan = crate::sparse_mapped_cost::build_unfolded(
            mapped.each_ref(),
            [indices_a, indices_b, indices_c],
            crate::sparse_mapped_cost::Inputs {
                storage,
                fractions: crate::sparse_cost::Fractions { a: 1., b: 1., c: 1. },
                custom: false,
            },
        ).expect("unsupported unfolded sparse raw mapping");
        assert_eq!(metadata.index_maps, plan.execution.indices);

        let rank = self.context().rank();
        let mut sparse_a = sparse_on_roots(a, &mapped[0]);
        let mut dense_b = dense_on_roots(b, &mapped[1]);
        let mut dense_c = dense_on_roots(self, &mapped[2]);
        let replication: [Vec<Context<'_>>; 3] = std::array::from_fn(|operand| {
            plan.execution.replication_axes[operand].iter()
                .map(|&axis| mapped[0].topology.fiber(self.context(), axis)).collect()
        });
        for communicator in &replication[0] {
            broadcast_sparse(communicator, &mut sparse_a);
        }
        for communicator in &replication[1] {
            communicator.broadcast(0, &mut dense_b);
        }
        let output_root = replication[2].iter().all(|communicator| communicator.rank() == 0);
        let child_beta = if output_root { beta } else { self.algebra().zero() };
        let dense_blocks = |values: Vec<A::Element>, block_size: usize| {
            assert!(block_size > 0 && values.len() % block_size == 0);
            values.chunks_exact(block_size).map(<[A::Element]>::to_vec).collect::<Vec<_>>()
        };
        let b_size: usize = plan.execution.block_shapes[1].iter().product();
        let c_size: usize = plan.execution.block_shapes[2].iter().product();
        let b_blocks = dense_blocks(dense_b, b_size);
        let c_blocks = dense_blocks(dense_c, c_size);
        let panel_contexts: Vec<[Option<Context<'_>>; 3]> = plan.execution.panels.iter()
            .map(|panel| std::array::from_fn(|operand| panel.operands[operand].topology_axis
                .map(|axis| mapped[0].topology.fiber(self.context(), axis)))).collect();
        let c_blocks = execute_raw_panels(
            &panel_contexts,
            &plan.execution,
            0,
            crate::sparse_2d::Layers { count: 1, index: 0 },
            self.algebra(),
            [indices_a, indices_b, indices_c],
            &alpha,
            &sparse_a,
            &b_blocks,
            c_blocks,
            child_beta,
        );
        for level in panel_contexts {
            for communicator in level.into_iter().flatten() { communicator.close(); }
        }
        dense_c = c_blocks.into_iter().flatten().collect();
        for communicator in &replication[2] {
            communicator.reduce_monoid(self.algebra(), &mut dense_c, commutative, 0);
        }
        let contributions: Vec<_> = dense_c.into_iter().enumerate().filter_map(|(offset, value)| {
            mapped[2].global_key(rank, offset)
                .filter(|&key| mapped[2].owner(key) == rank)
                .map(|key| (key, value))
        }).collect();
        for group in replication {
            for communicator in group { communicator.close(); }
        }
        let zero = self.algebra().zero();
        self.transform(|_, value| *value = zero.clone());
        self.write_add(&contributions);
    }

    /// Execute the exact sparse-A/dense-B/dense-C raw mapping and folded leaf
    /// selected by [`crate::sparse_search`]. No aligned-grid preparation or
    /// remapping fallback is performed.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_sparse_from_selected(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        selected: &crate::sparse_search::Selected,
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) {
        assert!(matches!(selected.pattern,
            crate::sparse_search::Pattern::SparseDenseDense { .. }));
        if selected.fold.is_none() {
            self.contract_sparse_from_mapped(indices_c, a, indices_a, b, indices_b,
                selected.distributions.clone(), alpha, beta, commutative);
            return;
        }
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        let mapped = &selected.distributions;
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, b.distribution().shape);
        assert_eq!(mapped[2].shape, self.distribution().shape);
        let descriptor = selected.fold.as_ref().unwrap();
        let plan = selected_plan(selected, [indices_a, indices_b, indices_c], [A::Element::WIDTH; 3], false);
        let rank = self.context().rank();
        let mut a = sparse_on_roots(a, &mapped[0]);
        let a_layout = folded_matricization(&plan.execution, descriptor, 0);
        let mut a: Vec<_> = a.drain(..)
            .map(|block| crate::sparse_matricize::matricize_pairs(&a_layout, &block)).collect();
        let virtual_blocks = mapped.each_ref().map(|distribution| distribution.mappings.iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase()).product());
        let mut b = descriptor.layouts[1].transpose(&dense_on_roots(b, &mapped[1]),
            virtual_blocks[1], crate::fold_layout::Direction::Forward);
        let mut c = descriptor.layouts[2].transpose(&dense_on_roots(self, &mapped[2]),
            virtual_blocks[2], crate::fold_layout::Direction::Forward);
        let replication: [Vec<Context<'_>>; 3] = std::array::from_fn(|operand| {
            plan.execution.replication_axes[operand].iter()
                .map(|&axis| mapped[0].topology.fiber(self.context(), axis)).collect()
        });
        replicate_coo(&replication[0], &mut a);
        for communicator in &replication[1] { communicator.broadcast(0, &mut b); }
        let output_root = replication[2].iter().all(|context| context.rank() == 0);
        let child_beta = if output_root { beta } else { self.algebra().zero() };
        let b_blocks = dense_blocks(b, descriptor.k * descriptor.n * descriptor.batches);
        let c_blocks = dense_blocks(c, descriptor.m * descriptor.n * descriptor.batches);
        let panel_contexts = panels(self.context(), &mapped[0].topology, &plan.execution);
        let c_blocks = match selected.pattern {
            crate::sparse_search::Pattern::SparseDenseDense { coo_kernel: true } => {
                execute_coo_dense_panels(&panel_contexts, &plan.execution, 0,
                    crate::sparse_2d::Layers { count: 1, index: 0 }, descriptor,
                    self.algebra(), &alpha, &a, &b_blocks, c_blocks, child_beta)
            }
            crate::sparse_search::Pattern::SparseDenseDense { coo_kernel: false } => {
                let a: Vec<_> = a.iter().map(Coo::to_csr).collect();
                execute_csr_dense_panels(&panel_contexts, &plan.execution, 0,
                    crate::sparse_2d::Layers { count: 1, index: 0 }, descriptor,
                    self.algebra(), &alpha, &a, &b_blocks, c_blocks, child_beta)
            }
            _ => unreachable!(),
        };
        close_contexts(panel_contexts);
        c = c_blocks.into_iter().flatten().collect();
        for communicator in &replication[2] {
            communicator.reduce_monoid(self.algebra(), &mut c, commutative, 0);
        }
        c = descriptor.layouts[2].transpose(&c, virtual_blocks[2],
            crate::fold_layout::Direction::Backward);
        let contributions: Vec<_> = c.into_iter().enumerate().filter_map(|(offset, value)| {
            mapped[2].global_key(rank, offset)
                .filter(|&key| mapped[2].owner(key) == rank).map(|key| (key, value))
        }).collect();
        for group in replication { for communicator in group { communicator.close(); } }
        let zero = self.algebra().zero();
        self.transform(|_, value| *value = zero.clone());
        self.write_add(&contributions);
    }

    /// Execute the exact folded sparse-A/sparse-B/dense-C selection.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_sparse_sparse_from_selected(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &SparseTensor<'_, '_, A>,
        indices_b: &str,
        selected: &crate::sparse_search::Selected,
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) {
        assert_eq!(selected.pattern, crate::sparse_search::Pattern::SparseSparseDense);
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        let mapped = &selected.distributions;
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, b.distribution().shape);
        assert_eq!(mapped[2].shape, self.distribution().shape);
        let descriptor = selected.fold.as_ref().expect("selected sparse-sparse path must be folded");
        let plan = selected_plan(selected, [indices_a, indices_b, indices_c], [A::Element::WIDTH; 3], false);
        let layouts = [folded_matricization(&plan.execution, descriptor, 0),
            folded_matricization(&plan.execution, descriptor, 1)];
        let mut aa: Vec<_> = sparse_on_roots(a, &mapped[0]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[0], &block)).collect();
        let mut bb: Vec<_> = sparse_on_roots(b, &mapped[1]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[1], &block)).collect();
        let virtual_blocks = mapped.each_ref().map(|distribution| distribution.mappings.iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase()).product());
        let mut c = descriptor.layouts[2].transpose(&dense_on_roots(self, &mapped[2]),
            virtual_blocks[2], crate::fold_layout::Direction::Forward);
        let replication: [Vec<Context<'_>>; 3] = std::array::from_fn(|operand| {
            plan.execution.replication_axes[operand].iter()
                .map(|&axis| mapped[0].topology.fiber(self.context(), axis)).collect()
        });
        replicate_coo(&replication[0], &mut aa);
        replicate_coo(&replication[1], &mut bb);
        let output_root = replication[2].iter().all(|context| context.rank() == 0);
        let child_beta = if output_root { beta } else { self.algebra().zero() };
        let aa: Vec<_> = aa.iter().map(Coo::to_csr).collect();
        let bb: Vec<_> = bb.iter().map(Coo::to_csr).collect();
        let c_blocks = dense_blocks(c, descriptor.m * descriptor.n * descriptor.batches);
        let panel_contexts = panels(self.context(), &mapped[0].topology, &plan.execution);
        let c_blocks = execute_csr_sparse_dense_panels(&panel_contexts, &plan.execution, 0,
            crate::sparse_2d::Layers { count: 1, index: 0 }, descriptor,
            self.algebra(), &alpha, &aa, &bb, c_blocks, child_beta);
        close_contexts(panel_contexts);
        c = c_blocks.into_iter().flatten().collect();
        for communicator in &replication[2] {
            communicator.reduce_monoid(self.algebra(), &mut c, commutative, 0);
        }
        c = descriptor.layouts[2].transpose(&c, virtual_blocks[2],
            crate::fold_layout::Direction::Backward);
        let rank = self.context().rank();
        let contributions: Vec<_> = c.into_iter().enumerate().filter_map(|(offset, value)| {
            mapped[2].global_key(rank, offset)
                .filter(|&key| mapped[2].owner(key) == rank).map(|key| (key, value))
        }).collect();
        for group in replication { for communicator in group { communicator.close(); } }
        let zero = self.algebra().zero();
        self.transform(|_, value| *value = zero.clone());
        self.write_add(&contributions);
    }

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

impl<A: Semiring + Clone> SparseTensor<'_, '_, A>
where
    A::Element: Wire,
{
    /// Eliminate sparse Hadamard indices before selecting and executing the
    /// sparse-A/sparse-B/sparse-C contraction.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_sparse(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &SparseTensor<'_, '_, A>,
        indices_b: &str,
        cache: &mut crate::sparse_search::SearchCache<'_, '_>,
        alpha: A::Element,
        beta: A::Element,
        output_fraction: Option<f64>,
        commutative: bool,
    ) -> Result<(), crate::sparse_search::Error> {
        let nonzeros = [canonical_nnz(a), canonical_nnz(b)];
        let distributions = [a.distribution(), b.distribution(), self.distribution()];
        let indices = [indices_a, indices_b, indices_c];
        if let Some(weigh) = sparse_weigh_index(
            distributions,
            indices,
            nonzeros.map(Some),
        ) {
            return eliminate_hadamard_index(a, indices_a, b, indices_b, weigh,
                |a, indices_a, b, indices_b| self.contract_sparse(
                    indices_c, a, indices_a, b, indices_b,
                    cache, alpha, beta, output_fraction, commutative,
                ));
        }
        let selected = cache.prepare(
            distributions,
            indices,
            [Some(nonzeros[0]), Some(nonzeros[1]), Some(canonical_nnz(self))],
            output_fraction,
        )?.expect("sparse contraction search found no eligible plan").clone();
        self.contract_sparse_from_selected(
            indices_c,
            a,
            indices_a,
            b,
            indices_b,
            &selected,
            alpha,
            beta,
            commutative,
        );
        Ok(())
    }

    /// Execute the exact folded sparse-A/sparse-B/sparse-C selection. The old
    /// sparse output is scaled once before the execution tree, then added only
    /// after the distributed product has been reduced.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_sparse_from_selected(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, A>,
        indices_a: &str,
        b: &SparseTensor<'_, '_, A>,
        indices_b: &str,
        selected: &crate::sparse_search::Selected,
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) {
        assert_eq!(selected.pattern, crate::sparse_search::Pattern::SparseSparseSparse);
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        let mapped = &selected.distributions;
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, b.distribution().shape);
        assert_eq!(mapped[2].shape, self.distribution().shape);
        let descriptor = selected.fold.as_ref().expect("selected sparse output path must be folded");
        let plan = selected_plan(selected, [indices_a, indices_b, indices_c], [A::Element::WIDTH; 3], false);
        let layouts = std::array::from_fn::<_, 3, _>(|operand| {
            folded_matricization(&plan.execution, descriptor, operand)
        });
        let mut aa: Vec<_> = sparse_on_roots(a, &mapped[0]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[0], &block)).collect();
        let mut bb: Vec<_> = sparse_on_roots(b, &mapped[1]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[1], &block)).collect();
        let mut old: Vec<_> = sparse_on_roots(self, &mapped[2]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[2], &block).to_csr())
            .collect();
        for block in &mut old {
            for value in block.values_mut() { *value = self.algebra().multiply(&beta, value); }
        }
        let replication: [Vec<Context<'_>>; 3] = std::array::from_fn(|operand| {
            plan.execution.replication_axes[operand].iter()
                .map(|&axis| mapped[0].topology.fiber(self.context(), axis)).collect()
        });
        replicate_coo(&replication[0], &mut aa);
        replicate_coo(&replication[1], &mut bb);
        let aa: Vec<_> = aa.iter().map(Coo::to_csr).collect();
        let bb: Vec<_> = bb.iter().map(Coo::to_csr).collect();
        let mut product: Vec<_> = old.iter().map(|block| {
            let shape = block.shape();
            Coo::new(shape.0, shape.1, Vec::new()).to_csr()
        }).collect();
        let panel_contexts = panels(self.context(), &mapped[0].topology, &plan.execution);
        product = execute_csr_panels(&panel_contexts, &plan.execution, 0,
            crate::sparse_2d::Layers { count: 1, index: 0 }, descriptor,
            self.algebra(), &alpha, &aa, &bb, product, self.algebra().zero());
        close_contexts(panel_contexts);
        reduce_csr(&replication[2], self.algebra(), &mut product);
        let output_root = replication[2].iter().all(|context| context.rank() == 0);
        if output_root {
            for (product, old) in product.iter_mut().zip(&old) {
                *product = old.add(product, self.algebra());
            }
        }
        let dematricization = folded_dematricization(&plan.execution, descriptor);
        let pinned: Vec<_> = product.iter().map(|block| {
            crate::sparse_matricize::dematricize_pairs(&dematricization, &block.to_coo())
        }).collect();
        let blocks = crate::sparse_keys::depin_output(
            &local_key_metadata(&mapped[2], self.context().rank()), &pinned);
        for group in replication { for communicator in group { communicator.close(); } }
        let original = self.distribution.clone();
        self.distribution = mapped[2].clone();
        self.blocks = blocks;
        self.redistribute(original);
        let _ = commutative;
    }
}

impl<C: Monoid + Clone> SparseTensor<'_, '_, C>
where
    C::Element: Wire,
{
    /// Eliminate sparse Hadamard indices before selecting and executing the
    /// heterogeneous sparse contraction, retaining its custom functions.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_sparse_function<AA, BB, F, G>(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, AA>,
        indices_a: &str,
        b: &SparseTensor<'_, '_, BB>,
        indices_b: &str,
        cache: &mut crate::sparse_search::SearchCache<'_, '_>,
        output_fraction: Option<f64>,
        function: &F,
        accumulate: &G,
    ) -> Result<(), crate::sparse_search::Error>
    where
        AA: Monoid + Clone,
        BB: Monoid + Clone,
        AA::Element: Wire,
        BB::Element: Wire,
        F: Fn(&AA::Element, &BB::Element) -> C::Element,
        G: Fn(C::Element, &mut C::Element),
    {
        let nonzeros = [canonical_nnz(a), canonical_nnz(b)];
        let distributions = [a.distribution(), b.distribution(), self.distribution()];
        let indices = [indices_a, indices_b, indices_c];
        if let Some(weigh) = sparse_weigh_index(
            distributions,
            indices,
            nonzeros.map(Some),
        ) {
            return eliminate_hadamard_index(a, indices_a, b, indices_b, weigh,
                |a, indices_a, b, indices_b| self.contract_sparse_function(
                    indices_c, a, indices_a, b, indices_b,
                    cache, output_fraction, function, accumulate,
                ));
        }
        let selected = cache.prepare(
            distributions,
            indices,
            [Some(nonzeros[0]), Some(nonzeros[1]), Some(canonical_nnz(self))],
            output_fraction,
        )?.expect("sparse contraction search found no eligible plan").clone();
        self.contract_sparse_function_from_selected(
            indices_c,
            a,
            indices_a,
            b,
            indices_b,
            &selected,
            function,
            accumulate,
        );
        Ok(())
    }

    /// Execute the exact folded heterogeneous sparse/sparse/sparse selection.
    /// The selected raw distributions, replication, panels and virtual tree are
    /// retained. `function` creates one path contribution and `accumulate`
    /// combines intersecting paths; the output monoid performs MPI reduction
    /// and the final old-C/product sparse addition. Custom coefficients are the
    /// source identities, so no synthetic multiplication is required on C.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_sparse_function_from_selected<AA, BB, F, G>(
        &mut self,
        indices_c: &str,
        a: &SparseTensor<'_, '_, AA>,
        indices_a: &str,
        b: &SparseTensor<'_, '_, BB>,
        indices_b: &str,
        selected: &crate::sparse_search::Selected,
        function: F,
        accumulate: G,
    )
    where
        AA: Monoid,
        BB: Monoid,
        AA::Element: Wire,
        BB::Element: Wire,
        F: Fn(&AA::Element, &BB::Element) -> C::Element,
        G: Fn(C::Element, &mut C::Element),
    {
        assert_eq!(selected.pattern, crate::sparse_search::Pattern::SparseSparseSparse);
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        let mapped = &selected.distributions;
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, b.distribution().shape);
        assert_eq!(mapped[2].shape, self.distribution().shape);
        let descriptor = selected.fold.as_ref()
            .expect("selected heterogeneous sparse output path must be folded");
        let plan = selected_plan(
            selected,
            [indices_a, indices_b, indices_c],
            [AA::Element::WIDTH, BB::Element::WIDTH, C::Element::WIDTH],
            true,
        );
        let layouts = std::array::from_fn::<_, 3, _>(|operand| {
            folded_matricization(&plan.execution, descriptor, operand)
        });
        let mut aa: Vec<_> = sparse_on_roots(a, &mapped[0]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[0], &block)).collect();
        let mut bb: Vec<_> = sparse_on_roots(b, &mapped[1]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[1], &block)).collect();
        let old: Vec<_> = sparse_on_roots(self, &mapped[2]).into_iter()
            .map(|block| crate::sparse_matricize::matricize_pairs(&layouts[2], &block).to_csr())
            .collect();
        let replication: [Vec<Context<'_>>; 3] = std::array::from_fn(|operand| {
            plan.execution.replication_axes[operand].iter()
                .map(|&axis| mapped[0].topology.fiber(self.context(), axis)).collect()
        });
        replicate_coo(&replication[0], &mut aa);
        replicate_coo(&replication[1], &mut bb);
        let aa: Vec<_> = aa.iter().map(Coo::to_csr).collect();
        let bb: Vec<_> = bb.iter().map(Coo::to_csr).collect();
        let mut product: Vec<_> = old.iter().map(crate::sparse_2d::csr_empty).collect();
        let panel_contexts = panels(self.context(), &mapped[0].topology, &plan.execution);
        product = execute_custom_csr_panels(
            &panel_contexts,
            &plan.execution,
            0,
            crate::sparse_2d::Layers { count: 1, index: 0 },
            descriptor,
            self.algebra(),
            &aa,
            &bb,
            product,
            &function,
            &accumulate,
        );
        close_contexts(panel_contexts);
        for communicator in &replication[2] {
            for block in &mut product {
                let shape = block.shape();
                *block = block.reduce(communicator, 0, self.algebra()).unwrap_or_else(|| {
                    Coo::new(shape.0, shape.1, Vec::new()).to_csr()
                });
            }
        }
        let output_root = replication[2].iter().all(|context| context.rank() == 0);
        if output_root {
            for (product, old) in product.iter_mut().zip(&old) {
                *product = old.add(product, self.algebra());
            }
        }
        let dematricization = folded_dematricization(&plan.execution, descriptor);
        let pinned: Vec<_> = product.iter().map(|block| {
            crate::sparse_matricize::dematricize_pairs(&dematricization, &block.to_coo())
        }).collect();
        let blocks = crate::sparse_keys::depin_output(
            &local_key_metadata(&mapped[2], self.context().rank()), &pinned);
        for group in replication { for communicator in group { communicator.close(); } }
        let original = self.distribution.clone();
        self.distribution = mapped[2].clone();
        self.blocks = blocks;
        self.redistribute(original);
    }
}
