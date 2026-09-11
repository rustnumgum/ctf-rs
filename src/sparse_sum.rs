// Adapted from cc4s CTF sparse summation key reindexing responsibilities.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Monoid, Semiring, Wire}, context::Context, diagonal::Projection,
    mapping::{Distribution, Mapping}, sparse::SparseTensor, tensor::Tensor};

pub(super) fn mapped_sparse_blocks<A: Monoid>(source: &SparseTensor<'_, '_, A>,
    target: &Distribution, pin: bool) -> Vec<Vec<(usize, A::Element)>>
where A::Element: Wire {
    let context = source.context();
    let rank = context.rank();
    let width = 8 + A::Element::WIDTH;
    let mut buckets = vec![Vec::new(); context.size()];
    for (key, value) in source.local_pairs() {
        if source.distribution().owner(key) != rank { continue; }
        u64::try_from(key).unwrap().encode(&mut buckets[target.owner(key)]);
        value.encode(&mut buckets[target.owner(key)]);
    }
    let block_size: usize = target.block_shape().iter().product();
    let block_count: usize = target.mappings.iter()
        .map(|mapping| mapping.phase() / mapping.physical_phase()).product();
    let mut blocks = vec![Vec::new(); block_count];
    for bytes in context.inner.exchange(&buckets) {
        for pair in bytes.chunks_exact(width) {
            let key = usize::try_from(u64::decode(&pair[..8])).unwrap();
            let block = target.local_offset(rank, key) / block_size;
            blocks[block].push((key, A::Element::decode(&pair[8..])));
        }
    }
    for block in &mut blocks { block.sort_by_key(|pair| pair.0); }
    if !pin { return blocks; }
    crate::sparse_keys::pin_blocks(&key_metadata(target, rank), &blocks)
}

pub(super) fn key_metadata(distribution: &Distribution, rank: usize) -> crate::sparse_keys::KeyMetadata {
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

fn mapped_dense<A: Semiring>(source: &Tensor<'_, '_, A>, target: &Distribution)
    -> Vec<A::Element> where A::Element: Wire {
    let rank = source.context().rank();
    let requests: Vec<_> = (0..target.local_len()).filter_map(|offset| {
        target.global_key(rank, offset).filter(|&key| target.owner(key) == rank)
            .map(|key| (offset, key))
    }).collect();
    let values = source.read(&requests.iter().map(|&(_, key)| key).collect::<Vec<_>>());
    let mut result = vec![source.algebra().zero(); target.local_len()];
    for ((offset, _), value) in requests.into_iter().zip(values) { result[offset] = value; }
    result
}

pub(super) fn broadcast_pairs<E: Wire + Clone>(context: &Context<'_>, root: usize,
    blocks: &mut Vec<Vec<(usize, E)>>) {
    let width = 8 + E::WIDTH;
    let mut sizes = vec![0u64; blocks.len()];
    if context.rank() == root {
        for (size, block) in sizes.iter_mut().zip(blocks.iter()) {
            *size = block.len().try_into().unwrap();
        }
    }
    context.broadcast(root, &mut sizes);
    let count: usize = sizes.iter().map(|&size| usize::try_from(size).unwrap()).sum();
    let mut bytes = Vec::with_capacity(count * width);
    if context.rank() == root {
        for (key, value) in blocks.iter().flatten() {
            u64::try_from(*key).unwrap().encode(&mut bytes);
            value.encode(&mut bytes);
        }
    } else {
        bytes.resize(count * width, 0);
    }
    context.inner.broadcast(root, &mut bytes);
    if context.rank() != root {
        let mut pairs = bytes.chunks_exact(width);
        *blocks = sizes.into_iter().map(|size| (0..size).map(|_| {
            let pair = pairs.next().unwrap();
            (usize::try_from(u64::decode(&pair[..8])).unwrap(), E::decode(&pair[8..]))
        }).collect()).collect();
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape.iter().map(|&length| { let old = stride; stride *= length; old }).collect()
}

pub(super) fn normalized_indices(indices: [&str; 2]) -> [Vec<usize>; 2] {
    let mut labels = Vec::new();
    std::array::from_fn(|operand| indices[operand].bytes().map(|label| {
        if let Some(id) = labels.iter().position(|&old| old == label) { id }
        else { labels.push(label); labels.len() - 1 }
    }).collect())
}

pub(super) fn permute_input_key(mut key: usize, shape: &[usize], permutation: &[Option<usize>]) -> usize {
    assert_eq!(shape.len(), permutation.len());
    let dimensions = permutation.iter().flatten().copied().max().map_or(0, |axis| axis + 1);
    let mut target_shape = vec![0; dimensions];
    for (axis, position) in permutation.iter().enumerate() {
        if let Some(position) = position { target_shape[*position] = shape[axis]; }
    }
    let target_strides = strides(&target_shape);
    let mut result = 0;
    for (axis, position) in permutation.iter().enumerate() {
        let coordinate = key % shape[axis];
        key /= shape[axis];
        if let Some(position) = position { result += coordinate * target_strides[*position]; }
    }
    result
}

pub(super) fn permute_output_key(mut key: usize, shape: &[usize], permutation: &[usize]) -> usize {
    assert_eq!(shape.len(), permutation.len());
    let mut target_shape = vec![0; shape.len()];
    for (axis, &position) in permutation.iter().enumerate() { target_shape[position] = shape[axis]; }
    let target_strides = strides(&target_shape);
    let mut result = 0;
    for (axis, &position) in permutation.iter().enumerate() {
        result += (key % shape[axis]) * target_strides[position];
        key /= shape[axis];
    }
    result
}

pub(super) fn restore_output_key(mut key: usize, shape: &[usize], permutation: &[usize]) -> usize {
    let mut target_shape = vec![0; shape.len()];
    for (axis, &position) in permutation.iter().enumerate() { target_shape[position] = shape[axis]; }
    let original_strides = strides(shape);
    let mut result = 0;
    for (position, &length) in target_shape.iter().enumerate() {
        let coordinate = key % length;
        key /= length;
        let axis = permutation.iter().position(|&candidate| candidate == position).unwrap();
        result += coordinate * original_strides[axis];
    }
    result
}

fn permute_dense<E: Clone>(values: &[E], block_shape: &[usize], blocks: usize,
    permutation: &[usize], forward: bool) -> Vec<E> {
    let block_size: usize = block_shape.iter().product();
    assert_eq!(values.len(), block_size * blocks);
    let mut result = values.to_vec();
    for block in 0..blocks {
        for key in 0..block_size {
            let target = if forward { permute_output_key(key, block_shape, permutation) }
                else { restore_output_key(key, block_shape, permutation) };
            result[block * block_size + target] = values[block * block_size + key].clone();
        }
    }
    result
}

fn merge_pairs<A: Semiring>(algebra: &A, old: &[(usize, A::Element)],
    additions: &[(usize, A::Element)]) -> Vec<(usize, A::Element)> {
    let mut result = Vec::with_capacity(old.len() + additions.len());
    let (mut i, mut j) = (0, 0);
    while i < old.len() || j < additions.len() {
        if j == additions.len() || (i < old.len() && old[i].0 < additions[j].0) {
            result.push(old[i].clone()); i += 1;
        } else if i == old.len() || additions[j].0 < old[i].0 {
            let key = additions[j].0;
            let mut value = additions[j].1.clone(); j += 1;
            while j < additions.len() && additions[j].0 == key {
                value = algebra.add(&value, &additions[j].1); j += 1;
            }
            result.push((key, value));
        } else {
            let key = old[i].0;
            let mut value = algebra.add(&additions[j].1, &old[i].1);
            i += 1;
            j += 1;
            while j < additions.len() && additions[j].0 == key {
                value = algebra.add(&value, &additions[j].1); j += 1;
            }
            result.push((key, value));
        }
    }
    result
}

struct Indices {
    input: Projection,
    output: Projection,
    sources: Vec<Option<usize>>,
    broadcasts: usize,
}
impl Indices {
    fn new(input: &Distribution, labels_a: &str, output: &Distribution, labels_b: &str) -> Self {
        let input = Projection::new(&input.shape, labels_a);
        let output = Projection::new(&output.shape, labels_b);
        let mut broadcasts = 1;
        let sources = output.labels.bytes().enumerate().map(|(axis, label)| {
            let source = input.labels.bytes().position(|candidate| candidate == label);
            if let Some(source) = source { assert_eq!(input.shape[source], output.shape[axis]); }
            else { broadcasts *= output.shape[axis]; }
            source
        }).collect();
        Self { input, output, sources, broadcasts }
    }
    fn contributions<E: Clone>(&self, a: &Distribution, b: &Distribution,
        pairs: Vec<(usize, E)>) -> Vec<(usize, E)> {
        let mut result = Vec::new();
        for (key, value) in pairs {
            let Some(input) = self.input.project(&a.decode_key(key)) else { continue; };
            for mut broadcast in 0..self.broadcasts {
                let coordinates: Vec<_> = self.sources.iter().enumerate().map(|(axis, source)| {
                    if let Some(source) = source { input[*source] }
                    else {
                        let coordinate = broadcast % self.output.shape[axis];
                        broadcast /= self.output.shape[axis];
                        coordinate
                    }
                }).collect();
                result.push((b.encode_key(&self.output.expand(&coordinates)), value.clone()));
            }
        }
        result
    }
}

impl<A: Semiring> SparseTensor<'_, '_, A> where A::Element: Wire {
    /// Indexed sparse summation with explicit destination distribution.
    pub fn sum_from(&mut self, indices_b: &str, a: &Self, indices_a: &str,
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()));
        let indices = Indices::new(a.distribution(), indices_a, self.distribution(), indices_b);
        let pairs: Vec<_> = a.local_pairs().into_iter()
            .filter(|(key, _)| a.distribution().owner(*key) == a.context().rank())
            .map(|(key, value)| (key, self.algebra().multiply(&value, &alpha))).collect();
        let contributions = indices.contributions(a.distribution(), self.distribution(), pairs);
        // Only the indexed diagonal of B participates when labels repeat.
        self.scale_indexed(indices_b, &beta);
        self.write_add(&contributions);
    }

    /// Dense-to-sparse indexed summation. The source dense tensor is traversed
    /// locally; sparse output is never allocated as a dense staging tensor.
    pub fn sum_from_dense(&mut self, indices_b: &str, a: &Tensor<'_, '_, A>, indices_a: &str,
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()));
        let indices = Indices::new(a.distribution(), indices_a, self.distribution(), indices_b);
        let pairs: Vec<_> = a.local_pairs().into_iter()
            .filter(|(key, _)| a.distribution().owner(*key) == a.context().rank())
            // High-level summation sparsifies dense A before the sparse merge.
            .filter(|(_, value)| *value != a.algebra().zero())
            .map(|(key, value)| (key, self.algebra().multiply(&value, &alpha))).collect();
        let contributions = indices.contributions(a.distribution(), self.distribution(), pairs);
        self.scale_indexed(indices_b, &beta);
        self.write_add(&contributions);
    }

    /// Execute a selected raw sparse-to-sparse summation. Sparse keys are
    /// pinned and permuted exactly as recorded by the selected source tree;
    /// output-only labels are expanded locally before the sorted sparse merge.
    #[allow(clippy::too_many_arguments)]
    pub fn sum_sparse_from_selected(&mut self, indices_b: &str, a: &Self, indices_a: &str,
        selected: &crate::sparse_sum_search::Selected,
        alpha: A::Element, beta: A::Element) {
        assert_eq!(selected.pattern, crate::sparse_sum_search::Pattern::SparseSparse);
        assert!(std::ptr::eq(self.context(), a.context()));
        let mapped = &selected.distributions;
        let execution = &selected.execution;
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, self.distribution().shape);
        assert_eq!(execution.block_shapes, mapped.each_ref().map(Distribution::block_shape));
        assert_eq!(execution.indices, normalized_indices([indices_a, indices_b]));
        assert!(execution.replication_axes[1].is_empty(),
            "source sparse output summation cannot reduce replicated output");
        let rank = self.context().rank();
        let mut input = mapped_sparse_blocks(a, &mapped[0], execution.pin_keys[0]);
        for block in &mut input {
            for pair in block.iter_mut() {
                pair.0 = permute_input_key(pair.0, &execution.block_shapes[0],
                    &execution.input_permutation);
            }
            block.sort_by_key(|pair| pair.0);
        }
        let mut output = mapped_sparse_blocks(self, &mapped[1], execution.pin_keys[1]);
        for block in &mut output {
            for pair in block.iter_mut() {
                pair.0 = permute_output_key(pair.0, &execution.block_shapes[1],
                    &execution.output_permutation);
                pair.1 = self.algebra().multiply(&pair.1, &beta);
            }
            block.sort_by_key(|pair| pair.0);
        }
        let replication: [Vec<Context<'_>>; 2] = std::array::from_fn(|operand| {
            execution.replication_axes[operand].iter()
                .map(|&axis| mapped[0].topology.fiber(self.context(), axis)).collect()
        });
        for communicator in &replication[0] { broadcast_pairs(communicator, 0, &mut input); }
        let one = self.algebra().one();
        crate::sparse_virtual::execute(
            &execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice),
            &false,
            &true,
            |blocks, _| {
                let mut additions = Vec::with_capacity(
                    input[blocks[0]].len() * execution.output_prefix);
                for (key, value) in &input[blocks[0]] {
                    let value = if alpha == one { value.clone() }
                        else { self.algebra().multiply(value, &alpha) };
                    for mapped_axis in 0..execution.output_prefix {
                        additions.push((key * execution.output_prefix + mapped_axis, value.clone()));
                    }
                }
                additions.sort_by_key(|pair| pair.0);
                output[blocks[1]] = merge_pairs(self.algebra(), &output[blocks[1]], &additions);
            },
        );
        for block in &mut output {
            for pair in block.iter_mut() {
                pair.0 = restore_output_key(pair.0, &execution.block_shapes[1],
                    &execution.output_permutation);
            }
            block.sort_by_key(|pair| pair.0);
        }
        if execution.pin_keys[1] {
            output = crate::sparse_keys::depin_output(&key_metadata(&mapped[1], rank), &output);
        }
        for group in replication { for communicator in group { communicator.close(); } }
        let original = self.distribution.clone();
        self.distribution = mapped[1].clone();
        self.blocks = output;
        self.redistribute(original);
    }
}

impl<A: Semiring + Clone> Tensor<'_, '_, A> where A::Element: Wire {
    pub fn sum_from_sparse(&mut self, indices_b: &str, a: &SparseTensor<'_, '_, A>, indices_a: &str,
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()));
        let indices = Indices::new(a.distribution(), indices_a, self.distribution(), indices_b);
        let pairs: Vec<_> = a.local_pairs().into_iter()
            .filter(|(key, _)| a.distribution().owner(*key) == a.context().rank())
            .map(|(key, value)| {
                let value = if indices.input.labels.is_empty() {
                    self.algebra().multiply(&alpha, &value)
                } else { self.algebra().multiply(&value, &alpha) };
                (key, value)
            }).collect();
        let contributions = indices.contributions(a.distribution(), self.distribution(), pairs);
        let algebra = self.algebra().clone();
        self.transform_indexed(indices_b, |value| *value = algebra.multiply(&beta, value));
        self.write_add(&contributions);
    }


    /// Execute a selected raw sparse-to-dense summation, including recorded
    /// sparse input replication, virtual traversal and dense output reduction.
    #[allow(clippy::too_many_arguments)]
    pub fn sum_sparse_from_selected(&mut self, indices_b: &str,
        a: &SparseTensor<'_, '_, A>, indices_a: &str,
        selected: &crate::sparse_sum_search::Selected,
        alpha: A::Element, beta: A::Element, commutative: bool) {
        assert_eq!(selected.pattern, crate::sparse_sum_search::Pattern::SparseDense);
        assert!(std::ptr::eq(self.context(), a.context()));
        let mapped = &selected.distributions;
        let execution = &selected.execution;
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, self.distribution().shape);
        assert_eq!(execution.block_shapes, mapped.each_ref().map(Distribution::block_shape));
        assert_eq!(execution.indices, normalized_indices([indices_a, indices_b]));
        let rank = self.context().rank();
        let mut input = mapped_sparse_blocks(a, &mapped[0], execution.pin_keys[0]);
        for block in &mut input {
            for pair in block.iter_mut() {
                pair.0 = permute_input_key(pair.0, &execution.block_shapes[0],
                    &execution.input_permutation);
            }
            block.sort_by_key(|pair| pair.0);
        }
        let virtual_blocks: usize = mapped[1].mappings.iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase()).product();
        let mut output = permute_dense(
            &mapped_dense(self, &mapped[1]),
            &execution.block_shapes[1],
            virtual_blocks,
            &execution.output_permutation,
            true,
        );
        let replication: [Vec<Context<'_>>; 2] = std::array::from_fn(|operand| {
            execution.replication_axes[operand].iter()
                .map(|&axis| mapped[0].topology.fiber(self.context(), axis)).collect()
        });
        for communicator in &replication[0] { broadcast_pairs(communicator, 0, &mut input); }
        let output_root = replication[1].iter().all(|communicator| communicator.rank() == 0);
        let zero = self.algebra().zero();
        if output_root {
            if beta == zero { output.fill(zero.clone()); }
            else if beta != self.algebra().one() {
                for value in &mut output { *value = self.algebra().multiply(&beta, value); }
            }
        } else {
            output.fill(zero.clone());
        }
        let block_size: usize = execution.block_shapes[1].iter().product();
        let mut output: Vec<_> = output.chunks_exact(block_size)
            .map(<[A::Element]>::to_vec).collect();
        let scalar_input = indices_a.is_empty();
        crate::sparse_virtual::execute(
            &execution.virtual_dimensions,
            execution.indices.each_ref().map(Vec::as_slice),
            &false,
            &true,
            |blocks, _| {
                for (key, value) in &input[blocks[0]] {
                    let value = if scalar_input {
                        self.algebra().multiply(&alpha, value)
                    } else {
                        self.algebra().multiply(value, &alpha)
                    };
                    for mapped_axis in 0..execution.output_prefix {
                        let at = key * execution.output_prefix + mapped_axis;
                        output[blocks[1]][at] = self.algebra().add(&value, &output[blocks[1]][at]);
                    }
                }
            },
        );
        let mut output: Vec<_> = output.into_iter().flatten().collect();
        for communicator in &replication[1] {
            communicator.reduce_monoid(self.algebra(), &mut output, commutative, 0);
        }
        output = permute_dense(
            &output,
            &execution.block_shapes[1],
            virtual_blocks,
            &execution.output_permutation,
            false,
        );
        let contributions: Vec<_> = output.into_iter().enumerate().filter_map(|(offset, value)| {
            mapped[1].global_key(rank, offset)
                .filter(|&key| mapped[1].owner(key) == rank).map(|key| (key, value))
        }).collect();
        for group in replication { for communicator in group { communicator.close(); } }
        let zero = self.algebra().zero();
        self.transform(|_, value| *value = zero.clone());
        self.write_add(&contributions);
    }
}
