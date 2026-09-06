// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted dense NS branch of summation/sym_seq_sum.cxx; packed symmetry and
// distributed tsum layers are separate responsibilities.
use crate::algebra::Semiring;
use crate::{algebra::Arithmetic,context::Context};

/// Native-double tsum_replicate layer: broadcast input blocks, retain the old
/// output only on reduction roots, execute the virtual layer, then all-reduce
/// output blocks in communicator order. Communicators are supplied explicitly.
pub fn replicated_f64(input_comms: &[&Context<'_>], output_comms: &[&Context<'_>],
    shape_a: &[usize], virtual_a: &[usize], indices_a: &str, a: &mut [f64],
    shape_b: &[usize], virtual_b: &[usize], indices_b: &str, b: &mut [f64],
    alpha: f64, beta: f64) {
    for comm in input_comms { comm.broadcast(0,a); }
    let root = output_comms.iter().all(|comm|comm.rank() == 0);
    if !root { b.fill(0.); }
    virtualized(&Arithmetic::<f64>::new(),shape_a,virtual_a,indices_a,a,
        shape_b,virtual_b,indices_b,b,&alpha,&if root {beta} else {1.});
    for comm in output_comms { comm.sum_f64(b); }
}

pub(crate) struct Indices { dimensions: Vec<usize>, offsets: Vec<Vec<usize>> }
impl Indices {
    pub(crate) fn new(operands: &[(&[usize], &str)]) -> Self {
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        for &(shape,indices) in operands {
            assert!(indices.is_ascii());
            assert_eq!(shape.len(),indices.len());
            for (&n,label) in shape.iter().zip(indices.bytes()) {
                if let Some(i) = labels.iter().position(|&l|l == label) { assert_eq!(dimensions[i],n); }
                else { labels.push(label); dimensions.push(n); }
            }
        }
        let offsets = operands.iter().map(|&(shape,indices)| {
            let mut offsets = vec![0;labels.len()];
            let mut stride = 1;
            for (&n,label) in shape.iter().zip(indices.bytes()) {
                let i = labels.iter().position(|&l|l == label).unwrap();
                offsets[i] += stride;
                stride *= n;
            }
            offsets
        }).collect();
        Self {dimensions,offsets}
    }
    pub(crate) fn for_each(&self, mut visit: impl FnMut(&[usize])) {
        if self.dimensions.contains(&0) { return; }
        let mut index = vec![0;self.dimensions.len()];
        let mut positions = vec![0;self.offsets.len()];
        loop {
            for (position,offsets) in positions.iter_mut().zip(&self.offsets) {
                *position = index.iter().zip(offsets).map(|(&i,&s)|i*s).sum();
            }
            visit(&positions);
            let mut axis = 0;
            while axis < index.len() {
                index[axis] += 1;
                if index[axis] < self.dimensions[axis] { break; }
                index[axis] = 0;
                axis += 1;
            }
            if axis == index.len() { break; }
        }
    }
}

/// Local nonsymmetric sequential sum, including diagonal index maps. Buffers
/// are column-major local blocks, not globally gathered distributed tensors.
pub fn sequential<A: Semiring>(algebra: &A, shape_a: &[usize], indices_a: &str, a: &[A::Element],
    shape_b: &[usize], indices_b: &str, b: &mut [A::Element], alpha: &A::Element, beta: &A::Element) {
    sequential_function(algebra,shape_a,indices_a,a,shape_b,indices_b,b,alpha,beta,Clone::clone);
}

/// Local virtual-block traversal from tsum_virt::run. Both block dimensions and
/// virtual phases must agree for shared index labels. Beta is applied only on
/// the first visit to each output block, including reductions over virtual axes.
pub fn virtualized<A: Semiring>(algebra: &A,
    shape_a: &[usize], virtual_a: &[usize], indices_a: &str, a: &[A::Element],
    shape_b: &[usize], virtual_b: &[usize], indices_b: &str, b: &mut [A::Element],
    alpha: &A::Element, beta: &A::Element) {
    assert_eq!(shape_a.len(),virtual_a.len());
    assert_eq!(shape_b.len(),virtual_b.len());
    assert!(virtual_a.iter().chain(virtual_b).all(|&n|n > 0));
    let block_a: usize = shape_a.iter().product();
    let block_b: usize = shape_b.iter().product();
    let count_a: usize = virtual_a.iter().product();
    let count_b: usize = virtual_b.iter().product();
    assert_eq!(a.len(),block_a*count_a);
    assert_eq!(b.len(),block_b*count_b);
    let blocks = Indices::new(&[(virtual_a,indices_a),(virtual_b,indices_b)]);
    let mut visited = vec![false;count_b];
    let one = algebra.one();
    blocks.for_each(|offsets| {
        let ia = offsets[0]; let ib = offsets[1];
        sequential(algebra,shape_a,indices_a,&a[ia*block_a..(ia+1)*block_a],
            shape_b,indices_b,&mut b[ib*block_b..(ib+1)*block_b],alpha,
            if visited[ib] {&one} else {beta});
        visited[ib] = true;
    });
}

/// Upstream custom-function ordering: transform the alpha-scaled input, then
/// accumulate. Beta is applied once per selected output, not once per reduction.
pub fn sequential_function<A: Semiring>(algebra: &A, shape_a: &[usize], indices_a: &str, a: &[A::Element],
    shape_b: &[usize], indices_b: &str, b: &mut [A::Element], alpha: &A::Element, beta: &A::Element,
    function: impl Fn(&A::Element) -> A::Element) {
    assert_eq!(a.len(),shape_a.iter().product());
    assert_eq!(b.len(),shape_b.iter().product());
    let space = Indices::new(&[(shape_a,indices_a),(shape_b,indices_b)]);
    let output = Indices::new(&[(shape_b,indices_b)]);
    output.for_each(|offsets| {
        let value = &mut b[offsets[0]];
        *value = if *beta == algebra.zero() {algebra.zero()} else {algebra.multiply(beta,value)};
    });
    space.for_each(|offsets| {
        let scaled = algebra.multiply(&a[offsets[0]],alpha);
        let value = function(&scaled);
        let output = &mut b[offsets[1]];
        *output = algebra.add(&value,output);
    });
}
