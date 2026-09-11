// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted dense NS branch of summation/sym_seq_sum.cxx; packed symmetry and
// distributed tsum layers are separate responsibilities.
use crate::algebra::Semiring;
use crate::{algebra::Arithmetic, context::Context};

/// General semiring version of tsum_replicate. MPI's user operation preserves
/// the declared monoid order; no root-side array gather is used.
pub fn replicated<A: Semiring>(
    algebra: &A,
    input_comms: &[&Context<'_>],
    output_comms: &[&Context<'_>],
    shape_a: &[usize],
    virtual_a: &[usize],
    indices_a: &str,
    a: &mut [A::Element],
    shape_b: &[usize],
    virtual_b: &[usize],
    indices_b: &str,
    b: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    commutative: bool,
) where
    A::Element: crate::algebra::Wire,
{
    for comm in input_comms {
        comm.broadcast(0, a);
    }
    let root = output_comms.iter().all(|comm| comm.rank() == 0);
    if !root {
        b.fill(algebra.zero());
    }
    let one = algebra.one();
    virtualized(
        algebra,
        shape_a,
        virtual_a,
        indices_a,
        a,
        shape_b,
        virtual_b,
        indices_b,
        b,
        alpha,
        if root { beta } else { &one },
    );
    for comm in output_comms {
        comm.all_reduce_monoid(algebra, b, commutative);
    }
}

/// Native-double tsum_replicate layer: broadcast input blocks, retain the old
/// output only on reduction roots, execute the virtual layer, then all-reduce
/// output blocks in communicator order. Communicators are supplied explicitly.
pub fn replicated_f64(
    input_comms: &[&Context<'_>],
    output_comms: &[&Context<'_>],
    shape_a: &[usize],
    virtual_a: &[usize],
    indices_a: &str,
    a: &mut [f64],
    shape_b: &[usize],
    virtual_b: &[usize],
    indices_b: &str,
    b: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    for comm in input_comms {
        comm.broadcast(0, a);
    }
    let root = output_comms.iter().all(|comm| comm.rank() == 0);
    if !root {
        b.fill(0.);
    }
    virtualized(
        &Arithmetic::<f64>::new(),
        shape_a,
        virtual_a,
        indices_a,
        a,
        shape_b,
        virtual_b,
        indices_b,
        b,
        &alpha,
        &if root { beta } else { 1. },
    );
    for comm in output_comms {
        comm.sum_f64(b);
    }
}

pub(crate) struct Indices {
    dimensions: Vec<usize>,
    offsets: Vec<Vec<usize>>,
}
impl Indices {
    pub(crate) fn new(operands: &[(&[usize], &str)]) -> Self {
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        for &(shape, indices) in operands {
            assert!(indices.is_ascii());
            assert_eq!(shape.len(), indices.len());
            for (&n, label) in shape.iter().zip(indices.bytes()) {
                if let Some(i) = labels.iter().position(|&l| l == label) {
                    assert_eq!(dimensions[i], n);
                } else {
                    labels.push(label);
                    dimensions.push(n);
                }
            }
        }
        let offsets = operands
            .iter()
            .map(|&(shape, indices)| {
                let mut offsets = vec![0; labels.len()];
                let mut stride = 1;
                for (&n, label) in shape.iter().zip(indices.bytes()) {
                    let i = labels.iter().position(|&l| l == label).unwrap();
                    offsets[i] += stride;
                    stride *= n;
                }
                offsets
            })
            .collect();
        Self {
            dimensions,
            offsets,
        }
    }
    pub(crate) fn for_each(&self, mut visit: impl FnMut(&[usize])) {
        if self.dimensions.contains(&0) {
            return;
        }
        let mut index = vec![0; self.dimensions.len()];
        let mut positions = vec![0; self.offsets.len()];
        loop {
            for (position, offsets) in positions.iter_mut().zip(&self.offsets) {
                *position = index.iter().zip(offsets).map(|(&i, &s)| i * s).sum();
            }
            visit(&positions);
            let mut axis = 0;
            while axis < index.len() {
                index[axis] += 1;
                if index[axis] < self.dimensions[axis] {
                    break;
                }
                index[axis] = 0;
                axis += 1;
            }
            if axis == index.len() {
                break;
            }
        }
    }
}

/// Local nonsymmetric sequential sum, including diagonal index maps. Buffers
/// are column-major local blocks, not globally gathered distributed tensors.
pub fn sequential<A: Semiring>(
    algebra: &A,
    shape_a: &[usize],
    indices_a: &str,
    a: &[A::Element],
    shape_b: &[usize],
    indices_b: &str,
    b: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
) {
    if let Some(plan) = FoldedSum::new(shape_a, indices_a, shape_b, indices_b) {
        plan.execute(algebra, a, b, alpha, beta);
        return;
    }
    sequential_function(
        algebra,
        shape_a,
        indices_a,
        a,
        shape_b,
        indices_b,
        b,
        alpha,
        beta,
        Clone::clone,
    );
}

struct FoldedSum {
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
    order_a: Vec<usize>,
    order_b: Vec<usize>,
    residual_shape_a: Vec<usize>,
    residual_shape_b: Vec<usize>,
    residual_indices_a: String,
    residual_indices_b: String,
    inner_stride: usize,
}

impl FoldedSum {
    fn new(shape_a: &[usize], indices_a: &str, shape_b: &[usize], indices_b: &str) -> Option<Self> {
        if !indices_a.is_ascii()
            || !indices_b.is_ascii()
            || shape_a.len() != indices_a.len()
            || shape_b.len() != indices_b.len()
        {
            return None;
        }
        for indices in [indices_a, indices_b] {
            for (axis, label) in indices.bytes().enumerate() {
                if indices.as_bytes()[..axis].contains(&label) {
                    return None;
                }
            }
        }

        let shared: Vec<_> = indices_a
            .bytes()
            .filter(|label| indices_b.as_bytes().contains(label))
            .collect();
        if shared.is_empty() {
            return None;
        }
        for label in &shared {
            let ia = indices_a
                .bytes()
                .position(|candidate| candidate == *label)
                .unwrap();
            let ib = indices_b
                .bytes()
                .position(|candidate| candidate == *label)
                .unwrap();
            if shape_a[ia] != shape_b[ib] {
                return None;
            }
        }

        let order = |indices: &str| {
            shared
                .iter()
                .copied()
                .chain(indices.bytes().filter(|label| !shared.contains(label)))
                .map(|label| {
                    indices
                        .bytes()
                        .position(|candidate| candidate == label)
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let order_a = order(indices_a);
        let order_b = order(indices_b);
        let residual_a: Vec<_> = indices_a
            .bytes()
            .enumerate()
            .filter(|(_, label)| !shared.contains(label))
            .collect();
        let residual_b: Vec<_> = indices_b
            .bytes()
            .enumerate()
            .filter(|(_, label)| !shared.contains(label))
            .collect();
        Some(Self {
            shape_a: shape_a.to_vec(),
            shape_b: shape_b.to_vec(),
            order_a,
            order_b,
            residual_shape_a: residual_a.iter().map(|(axis, _)| shape_a[*axis]).collect(),
            residual_shape_b: residual_b.iter().map(|(axis, _)| shape_b[*axis]).collect(),
            residual_indices_a: String::from_utf8(
                residual_a.iter().map(|(_, label)| *label).collect(),
            )
            .unwrap(),
            residual_indices_b: String::from_utf8(
                residual_b.iter().map(|(_, label)| *label).collect(),
            )
            .unwrap(),
            inner_stride: shared
                .iter()
                .map(|label| {
                    shape_a[indices_a
                        .bytes()
                        .position(|candidate| candidate == *label)
                        .unwrap()]
                })
                .product(),
        })
    }

    fn execute<A: Semiring>(
        &self,
        algebra: &A,
        a: &[A::Element],
        b: &mut [A::Element],
        alpha: &A::Element,
        beta: &A::Element,
    ) {
        assert_eq!(a.len(), self.shape_a.iter().product());
        assert_eq!(b.len(), self.shape_b.iter().product());
        let packed_a = pack(a, &self.shape_a, &self.order_a);
        let mut packed_b = pack(b, &self.shape_b, &self.order_b);

        if *beta == algebra.zero() {
            packed_b.fill(algebra.zero());
        } else if *beta != algebra.one() {
            for value in &mut packed_b {
                *value = algebra.multiply(beta, value);
            }
        }

        let residual = Indices::new(&[
            (&self.residual_shape_a, &self.residual_indices_a),
            (&self.residual_shape_b, &self.residual_indices_b),
        ]);
        residual.for_each(|offsets| {
            let start_a = offsets[0] * self.inner_stride;
            let start_b = offsets[1] * self.inner_stride;
            for inner in 0..self.inner_stride {
                let scaled = algebra.multiply(&packed_a[start_a + inner], alpha);
                packed_b[start_b + inner] = algebra.add(&scaled, &packed_b[start_b + inner]);
            }
        });
        unpack(&packed_b, b, &self.shape_b, &self.order_b);
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape
        .iter()
        .map(|&dimension| {
            let current = stride;
            stride *= dimension;
            current
        })
        .collect()
}

fn pack<T: Clone>(source: &[T], shape: &[usize], order: &[usize]) -> Vec<T> {
    let source_strides = strides(shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| shape[axis]).collect();
    let mut packed = source.to_vec();
    for (packed_offset, value) in packed.iter_mut().enumerate() {
        let mut remainder = packed_offset;
        let mut source_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            source_offset += remainder % dimension * source_strides[axis];
            remainder /= dimension;
        }
        *value = source[source_offset].clone();
    }
    packed
}

fn unpack<T: Clone>(packed: &[T], target: &mut [T], shape: &[usize], order: &[usize]) {
    let target_strides = strides(shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| shape[axis]).collect();
    for (packed_offset, value) in packed.iter().enumerate() {
        let mut remainder = packed_offset;
        let mut target_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            target_offset += remainder % dimension * target_strides[axis];
            remainder /= dimension;
        }
        target[target_offset] = value.clone();
    }
}

/// Local virtual-block traversal from tsum_virt::run. Both block dimensions and
/// virtual phases must agree for shared index labels. Beta is applied only on
/// the first visit to each output block, including reductions over virtual axes.
pub fn virtualized<A: Semiring>(
    algebra: &A,
    shape_a: &[usize],
    virtual_a: &[usize],
    indices_a: &str,
    a: &[A::Element],
    shape_b: &[usize],
    virtual_b: &[usize],
    indices_b: &str,
    b: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
) {
    assert_eq!(shape_a.len(), virtual_a.len());
    assert_eq!(shape_b.len(), virtual_b.len());
    assert!(virtual_a.iter().chain(virtual_b).all(|&n| n > 0));
    let block_a: usize = shape_a.iter().product();
    let block_b: usize = shape_b.iter().product();
    let count_a: usize = virtual_a.iter().product();
    let count_b: usize = virtual_b.iter().product();
    assert_eq!(a.len(), block_a * count_a);
    assert_eq!(b.len(), block_b * count_b);
    let blocks = Indices::new(&[(virtual_a, indices_a), (virtual_b, indices_b)]);
    let mut visited = vec![false; count_b];
    let one = algebra.one();
    blocks.for_each(|offsets| {
        let ia = offsets[0];
        let ib = offsets[1];
        sequential(
            algebra,
            shape_a,
            indices_a,
            &a[ia * block_a..(ia + 1) * block_a],
            shape_b,
            indices_b,
            &mut b[ib * block_b..(ib + 1) * block_b],
            alpha,
            if visited[ib] { &one } else { beta },
        );
        visited[ib] = true;
    });
}

/// Custom `tsum_virt` branch. The function sees the source alpha-scaled value,
/// and beta is applied on only the first visit to each output virtual block.
pub fn virtualized_function<A: Semiring>(
    algebra: &A,
    shape_a: &[usize],
    virtual_a: &[usize],
    indices_a: &str,
    a: &[A::Element],
    shape_b: &[usize],
    virtual_b: &[usize],
    indices_b: &str,
    b: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    function: &impl Fn(&A::Element) -> A::Element,
) {
    assert_eq!(shape_a.len(), virtual_a.len());
    assert_eq!(shape_b.len(), virtual_b.len());
    assert!(virtual_a.iter().chain(virtual_b).all(|&n| n > 0));
    let block_a: usize = shape_a.iter().product();
    let block_b: usize = shape_b.iter().product();
    let count_a: usize = virtual_a.iter().product();
    let count_b: usize = virtual_b.iter().product();
    assert_eq!(a.len(), block_a * count_a);
    assert_eq!(b.len(), block_b * count_b);
    let blocks = Indices::new(&[(virtual_a, indices_a), (virtual_b, indices_b)]);
    let mut visited = vec![false; count_b];
    let one = algebra.one();
    blocks.for_each(|offsets| {
        let ia = offsets[0];
        let ib = offsets[1];
        sequential_function(
            algebra,
            shape_a,
            indices_a,
            &a[ia * block_a..(ia + 1) * block_a],
            shape_b,
            indices_b,
            &mut b[ib * block_b..(ib + 1) * block_b],
            alpha,
            if visited[ib] { &one } else { beta },
            function,
        );
        visited[ib] = true;
    });
}

/// Distributed custom `seq_tsr_sum` branch. Custom execution stays non-inner,
/// matching the source restriction; only ordinary sums use `FoldedSum`.
pub fn replicated_function<A: Semiring>(
    algebra: &A,
    input_comms: &[&Context<'_>],
    output_comms: &[&Context<'_>],
    shape_a: &[usize],
    virtual_a: &[usize],
    indices_a: &str,
    a: &mut [A::Element],
    shape_b: &[usize],
    virtual_b: &[usize],
    indices_b: &str,
    b: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    commutative: bool,
    function: &impl Fn(&A::Element) -> A::Element,
) where
    A::Element: crate::algebra::Wire,
{
    for comm in input_comms {
        comm.broadcast(0, a);
    }
    let root = output_comms.iter().all(|comm| comm.rank() == 0);
    if !root {
        b.fill(algebra.zero());
    }
    let one = algebra.one();
    virtualized_function(
        algebra,
        shape_a,
        virtual_a,
        indices_a,
        a,
        shape_b,
        virtual_b,
        indices_b,
        b,
        alpha,
        if root { beta } else { &one },
        function,
    );
    for comm in output_comms {
        comm.all_reduce_monoid(algebra, b, commutative);
    }
}

/// Upstream custom-function ordering: transform the alpha-scaled input, then
/// accumulate. Beta is applied once per selected output, not once per reduction.
pub fn sequential_function<A: Semiring>(
    algebra: &A,
    shape_a: &[usize],
    indices_a: &str,
    a: &[A::Element],
    shape_b: &[usize],
    indices_b: &str,
    b: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    function: impl Fn(&A::Element) -> A::Element,
) {
    assert_eq!(a.len(), shape_a.iter().product());
    assert_eq!(b.len(), shape_b.iter().product());
    let space = Indices::new(&[(shape_a, indices_a), (shape_b, indices_b)]);
    let output = Indices::new(&[(shape_b, indices_b)]);
    output.for_each(|offsets| {
        let value = &mut b[offsets[0]];
        *value = if *beta == algebra.zero() {
            algebra.zero()
        } else {
            algebra.multiply(beta, value)
        };
    });
    space.for_each(|offsets| {
        let scaled = algebra.multiply(&a[offsets[0]], alpha);
        let value = function(&scaled);
        let output = &mut b[offsets[1]];
        *output = algebra.add(&value, output);
    });
}
