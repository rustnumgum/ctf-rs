// Adapted from cc4s CTF sparse unary functions and accumulator transforms.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Monoid, Wire}, context::Context, diagonal::Projection, mapping::Distribution,
    sparse::SparseTensor, tensor::Tensor};

trait SelectedUpdate<I, O> {
    fn missing(&self, input: &I) -> Option<O>;
    fn first_existing(&self, input: &I, output: &mut O);
    fn existing(&self, input: &I, output: &mut O);
}

struct Apply<'a, A: Monoid, F> {
    algebra: &'a A,
    function: F,
}

impl<I, A: Monoid, F: Fn(&I) -> A::Element> SelectedUpdate<I, A::Element> for Apply<'_, A, F> {
    fn missing(&self, input: &I) -> Option<A::Element> { Some((self.function)(input)) }
    fn first_existing(&self, input: &I, output: &mut A::Element) {
        *output = self.algebra.add(&(self.function)(input), output);
    }
    fn existing(&self, input: &I, output: &mut A::Element) {
        *output = self.algebra.add(output, &(self.function)(input));
    }
}

struct Accumulate<F> { function: F }

impl<I, O, F: Fn(&I, &mut O)> SelectedUpdate<I, O> for Accumulate<F> {
    fn missing(&self, _: &I) -> Option<O> { None }
    fn first_existing(&self, input: &I, output: &mut O) { (self.function)(input, output); }
    fn existing(&self, input: &I, output: &mut O) { (self.function)(input, output); }
}

fn merge_selected<I, O: Clone>(old: &[(usize, O)], additions: &[(usize, &I)],
    update: &impl SelectedUpdate<I, O>) -> Vec<(usize, O)> {
    let mut result = Vec::with_capacity(old.len() + additions.len());
    let (mut i, mut j) = (0, 0);
    while i < old.len() || j < additions.len() {
        if j == additions.len() || (i < old.len() && old[i].0 < additions[j].0) {
            result.push(old[i].clone());
            i += 1;
        } else if i == old.len() || additions[j].0 < old[i].0 {
            let key = additions[j].0;
            if let Some(mut value) = update.missing(additions[j].1) {
                j += 1;
                while j < additions.len() && additions[j].0 == key {
                    update.existing(additions[j].1, &mut value);
                    j += 1;
                }
                result.push((key, value));
            } else {
                while j < additions.len() && additions[j].0 == key { j += 1; }
            }
        } else {
            let key = old[i].0;
            let mut value = old[i].1.clone();
            i += 1;
            update.first_existing(additions[j].1, &mut value);
            j += 1;
            while j < additions.len() && additions[j].0 == key {
                update.existing(additions[j].1, &mut value);
                j += 1;
            }
            result.push((key, value));
        }
    }
    result
}

fn execute_selected_update<I, O, U>(output: &mut SparseTensor<'_, '_, O>,
    indices_b: &str, input: &SparseTensor<'_, '_, I>, indices_a: &str,
    selected: &crate::sparse_sum_search::Selected,
    update: U)
where
    I: Monoid,
    O: Monoid + Clone,
    I::Element: Wire,
    O::Element: Wire,
    U: SelectedUpdate<I::Element, O::Element>,
{
    assert_eq!(selected.pattern, crate::sparse_sum_search::Pattern::SparseSparse);
    assert!(std::ptr::eq(output.context(), input.context()));
    let mapped = &selected.distributions;
    let execution = &selected.execution;
    assert_eq!(mapped[0].shape, input.distribution().shape);
    assert_eq!(mapped[1].shape, output.distribution().shape);
    assert_eq!(execution.block_shapes, mapped.each_ref().map(Distribution::block_shape));
    assert_eq!(execution.indices, super::summation::normalized_indices([indices_a, indices_b]));
    assert!(execution.replication_axes[1].is_empty(),
        "source sparse output summation cannot reduce replicated output");
    let rank = output.context().rank();
    let mut a = super::summation::mapped_sparse_blocks(
        input, &mapped[0], execution.pin_keys[0]);
    for block in &mut a {
        for pair in block.iter_mut() {
            pair.0 = super::summation::permute_input_key(
                pair.0, &execution.block_shapes[0], &execution.input_permutation);
        }
        block.sort_by_key(|pair| pair.0);
    }
    let mut b = super::summation::mapped_sparse_blocks(
        output, &mapped[1], execution.pin_keys[1]);
    for block in &mut b {
        for pair in block.iter_mut() {
            pair.0 = super::summation::permute_output_key(
                pair.0, &execution.block_shapes[1], &execution.output_permutation);
        }
        block.sort_by_key(|pair| pair.0);
    }
    let replication: [Vec<Context<'_>>; 2] = std::array::from_fn(|operand| {
        execution.replication_axes[operand].iter()
            .map(|&axis| mapped[0].topology.fiber(output.context(), axis)).collect()
    });
    for communicator in &replication[0] {
        super::summation::broadcast_pairs(communicator, 0, &mut a);
    }
    crate::sparse_virtual::execute(
        &execution.virtual_dimensions,
        execution.indices.each_ref().map(Vec::as_slice),
        &false,
        &true,
        |blocks, _| {
            let mut additions = Vec::with_capacity(
                a[blocks[0]].len() * execution.output_prefix);
            for (key, value) in &a[blocks[0]] {
                for mapped_axis in 0..execution.output_prefix {
                    additions.push((key * execution.output_prefix + mapped_axis, value));
                }
            }
            additions.sort_by_key(|pair| pair.0);
            b[blocks[1]] = merge_selected(&b[blocks[1]], &additions, &update);
        },
    );
    for block in &mut b {
        for pair in block.iter_mut() {
            pair.0 = super::summation::restore_output_key(
                pair.0, &execution.block_shapes[1], &execution.output_permutation);
        }
        block.sort_by_key(|pair| pair.0);
    }
    if execution.pin_keys[1] {
        b = crate::sparse_keys::depin_output(
            &super::summation::key_metadata(&mapped[1], rank), &b);
    }
    for group in replication { for communicator in group { communicator.close(); } }
    let original = output.distribution.clone();
    output.distribution = mapped[1].clone();
    output.blocks = b;
    output.redistribute(original);
}

impl<'c, 'r, A: Monoid> SparseTensor<'c, 'r, A> {
    /// Apply a typed unary function to stored entries only. The distribution
    /// and explicit sparse structure are preserved, even when f(value) is zero.
    /// This operation is local and does not require an MPI wire format for B.
    pub fn map_stored<B: Monoid>(&self, algebra: B,
        mut function: impl FnMut(&A::Element) -> B::Element) -> SparseTensor<'c, 'r, B> {
        let mut result = SparseTensor::new(self.context(), self.distribution().clone(), algebra);
        for (target, source) in result.blocks.iter_mut().zip(&self.blocks) {
            target.extend(source.iter().map(|(key, value)| (*key, function(value))));
        }
        result
    }

    /// Accumulate a dense operand into existing sparse entries, without adding
    /// any new sparse keys. Input labels must occur in the output labels;
    /// repeated output labels restrict the transform to its indexed diagonal.
    /// Dense additive-identity values are skipped: upstream sparsifies dense A
    /// before executing its sparse accumulator kernel.
    /// Collective indexed reads communicate only input entries needed locally.
    pub fn accumulate_from_dense<I: Monoid>(&mut self, indices_b: &str,
        a: &Tensor<'_, '_, I>, indices_a: &str,
        mut function: impl FnMut(&I::Element, &mut A::Element)) where I::Element: Wire {
        assert!(std::ptr::eq(self.context(), a.context()));
        let (keys, positions) = self.accumulator_keys(indices_b, a.distribution(), indices_a);
        let values = a.read(&keys);
        for ((block, offset), value) in positions.into_iter().zip(values) {
            if value != a.algebra().zero() {
                function(&value, &mut self.blocks[block][offset].1);
            }
        }
    }

    /// Sparse-input accumulator: absent input entries do not invoke the
    /// function, while explicit stored zeros do. Output structure is unchanged.
    pub fn accumulate_from_sparse<I: Monoid>(&mut self, indices_b: &str,
        a: &SparseTensor<'_, '_, I>, indices_a: &str,
        mut function: impl FnMut(&I::Element, &mut A::Element)) where I::Element: Wire {
        assert!(std::ptr::eq(self.context(), a.context()));
        let (keys, positions) = self.accumulator_keys(indices_b, a.distribution(), indices_a);
        let mut requests = vec![Vec::new(); self.context().size()];
        for (position, &key) in keys.iter().enumerate() {
            let bucket = &mut requests[a.distribution().owner(key)];
            (position as u64).encode(bucket);
            (key as u64).encode(bucket);
        }
        let mut replies = vec![Vec::new(); self.context().size()];
        for (rank, bytes) in self.context().inner.exchange(&requests).iter().enumerate() {
            for request in bytes.chunks_exact(16) {
                let position = u64::decode(&request[..8]);
                let key = u64::decode(&request[8..]) as usize;
                let block = &a.blocks[a.block(key)];
                if let Ok(index) = block.binary_search_by_key(&key, |pair| pair.0) {
                    position.encode(&mut replies[rank]);
                    block[index].1.encode(&mut replies[rank]);
                }
            }
        }
        let mut values = vec![None; positions.len()];
        for bytes in self.context().inner.exchange(&replies) {
            for reply in bytes.chunks_exact(8 + I::Element::WIDTH) {
                let position = u64::decode(&reply[..8]) as usize;
                values[position] = Some(I::Element::decode(&reply[8..]));
            }
        }
        for ((block, offset), value) in positions.into_iter().zip(values) {
            if let Some(value) = value { function(&value, &mut self.blocks[block][offset].1); }
        }
    }

    fn accumulator_keys(&self, indices_b: &str, a: &Distribution, indices_a: &str)
        -> (Vec<usize>, Vec<(usize, usize)>) {
        let output = Projection::new(&self.distribution().shape, indices_b);
        let input = Projection::new(&a.shape, indices_a);
        let axes: Vec<_> = input.labels.bytes().enumerate().map(|(axis, label)| {
            let source = output.labels.bytes().position(|candidate| candidate == label)
                .expect("accumulator input labels must occur in output");
            assert_eq!(input.shape[axis], output.shape[source]);
            source
        }).collect();
        let mut keys = Vec::new();
        let mut positions = Vec::new();
        for (block, pairs) in self.blocks.iter().enumerate() {
            for (offset, (key, _)) in pairs.iter().enumerate() {
                let Some(coordinates) = output.project(&self.distribution().decode_key(*key))
                    else { continue; };
                let input_coordinates: Vec<_> = axes.iter().map(|&axis| coordinates[axis]).collect();
                keys.push(a.encode_key(&input.expand(&input_coordinates)));
                positions.push((block, offset));
            }
        }
        (keys, positions)
    }
}

impl<O: Monoid + Clone> SparseTensor<'_, '_, O>
where O::Element: Wire {
    /// Apply a heterogeneous unary function along an exact selected sparse sum.
    /// Missing output keys are created, then collisions use the output monoid.
    pub fn sum_sparse_function_from_selected<I: Monoid>(
        &mut self,
        indices_b: &str,
        input: &SparseTensor<'_, '_, I>,
        indices_a: &str,
        selected: &crate::sparse_sum_search::Selected,
        function: impl Fn(&I::Element) -> O::Element,
    ) where I::Element: Wire {
        let algebra = self.algebra().clone();
        execute_selected_update(self, indices_b, input, indices_a, selected, Apply {
            algebra: &algebra, function,
        });
    }

    /// Apply an accumulator only where selected sparse input and output
    /// structures intersect. Missing output keys are deliberately not created.
    pub fn accumulate_sparse_function_from_selected<I: Monoid>(
        &mut self,
        indices_b: &str,
        input: &SparseTensor<'_, '_, I>,
        indices_a: &str,
        selected: &crate::sparse_sum_search::Selected,
        function: impl Fn(&I::Element, &mut O::Element),
    ) where I::Element: Wire {
        execute_selected_update(self, indices_b, input, indices_a, selected,
            Accumulate { function });
    }
}
