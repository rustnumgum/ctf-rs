// Adapted from cc4s CTF contraction/contraction.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Local dense nonsymmetric contraction folding.
//!
//! The index grouping and matrix layout follow `get_len_ordering`,
//! `calc_fold_lnmk`, and all six dense permutations in CTF's
//! `src/contraction/contraction.cxx` at
//! `f69cbb46e23bc2f39cda5722ce096f56301dab4f`: AB, AC, BC, and ABC labels
//! become `k`, `m`, `n`, and independent GEMM batches `l`, respectively.

use crate::{
    contraction::{Folded, folded},
    linalg::{GemmKernel, Transpose},
};

const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [2, 0, 1],
    [1, 2, 0],
    [1, 0, 2],
    [2, 1, 0],
];
const TRANSPOSES: [(Transpose, Transpose, bool); 6] = [
    (Transpose::Yes, Transpose::No, false),
    (Transpose::No, Transpose::No, false),
    (Transpose::No, Transpose::Yes, false),
    (Transpose::No, Transpose::No, true),
    (Transpose::No, Transpose::Yes, true),
    (Transpose::Yes, Transpose::No, true),
];

/// Tensor named by a rejected contraction description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    A,
    B,
    C,
}

/// Reason that a local contraction cannot be represented by the bounded fold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rejected {
    NonAsciiIndices {
        operand: Operand,
    },
    RankMismatch {
        operand: Operand,
        shape: usize,
        indices: usize,
    },
    RepeatedIndex {
        operand: Operand,
        label: char,
    },
    DimensionMismatch {
        label: char,
    },
    OneOperandLabel {
        operand: Operand,
        label: char,
    },
    SizeOverflow,
}

/// A fully foldable local dense contraction.
///
/// Permutation zero packs `A[k,m,l]`, `B[k,n,l]`, `C[m,n,l]`; the selected
/// source permutation controls the other layouts and GEMM transpose flags.
/// Group labels keep original normalized order across all permutations.
#[derive(Clone, Debug)]
pub struct Plan {
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
    shape_c: Vec<usize>,
    order_a: Vec<usize>,
    order_b: Vec<usize>,
    order_c: Vec<usize>,
    m: usize,
    n: usize,
    k: usize,
    batches: usize,
    size_a: usize,
    size_b: usize,
    size_c: usize,
    permutation: usize,
    transpose_seconds: [f64; 3],
    trans_a: Transpose,
    trans_b: Transpose,
    transposed_output: bool,
}

impl Plan {
    /// Classify and plan a unique-index dense contraction.
    pub fn new(shapes: [&[usize]; 3], indices: [&str; 3]) -> Result<Self, Rejected> {
        Self::with_permutation(shapes, indices, 0)
    }

    /// Select the source dense fold permutation. Transpose ties deliberately
    /// select the last candidate, matching select_ctr_perm's `<=` comparison.
    pub fn select(
        shapes: [&[usize]; 3],
        indices: [&str; 3],
        models: &crate::cost::Models,
        virtual_copies: [usize; 3],
        commutative: bool,
    ) -> Result<Self, Rejected> {
        let count = if commutative { 6 } else { 3 };
        let mut selected = None;
        let mut selected_time = f64::MAX;
        for permutation in 0..count {
            let mut plan = Self::with_permutation(shapes, indices, permutation)?;
            let mut times = std::array::from_fn(|operand| {
                virtual_copies[operand]
                    as f64
                    * models.transpose(shapes[operand], plan.order(operand))
            });
            times[PERMUTATIONS[permutation][2]] *= 2.;
            let roles = PERMUTATIONS[permutation];
            let total = times[roles[0]] + times[roles[1]] + times[roles[2]];
            if total <= selected_time {
                selected_time = total;
                plan.transpose_seconds = times;
                selected = Some(plan);
            }
        }
        Ok(selected.unwrap())
    }

    pub fn permutation(&self) -> usize {
        self.permutation
    }

    pub fn transpose_seconds(&self) -> [f64; 3] {
        self.transpose_seconds
    }

    fn order(&self, operand: usize) -> &[usize] {
        match operand {
            0 => &self.order_a,
            1 => &self.order_b,
            2 => &self.order_c,
            _ => panic!("fold operand must be in 0..3"),
        }
    }

    fn with_permutation(
        shapes: [&[usize]; 3],
        indices: [&str; 3],
        permutation: usize,
    ) -> Result<Self, Rejected> {
        let roles = PERMUTATIONS[permutation];
        let operands = [Operand::A, Operand::B, Operand::C];
        for i in 0..3 {
            if !indices[i].is_ascii() {
                return Err(Rejected::NonAsciiIndices {
                    operand: operands[i],
                });
            }
            if shapes[i].len() != indices[i].len() {
                return Err(Rejected::RankMismatch {
                    operand: operands[i],
                    shape: shapes[i].len(),
                    indices: indices[i].len(),
                });
            }
            let mut seen = [false; 256];
            for label in indices[i].bytes() {
                if seen[label as usize] {
                    return Err(Rejected::RepeatedIndex {
                        operand: operands[i],
                        label: label as char,
                    });
                }
                seen[label as usize] = true;
            }
        }

        let mut masks = [0_u8; 256];
        let mut dimensions = [None; 256];
        for i in 0..3 {
            for (&dimension, label) in shapes[i].iter().zip(indices[i].bytes()) {
                if let Some(previous) = dimensions[label as usize] {
                    if previous != dimension {
                        return Err(Rejected::DimensionMismatch {
                            label: label as char,
                        });
                    }
                } else {
                    dimensions[label as usize] = Some(dimension);
                }
                masks[label as usize] |= 1 << i;
            }
        }
        for (label, &mask) in masks.iter().enumerate() {
            let operand = match mask {
                0 | 0b011 | 0b101 | 0b110 | 0b111 => continue,
                0b001 => Operand::A,
                0b010 => Operand::B,
                0b100 => Operand::C,
                _ => unreachable!(),
            };
            return Err(Rejected::OneOperandLabel {
                operand,
                label: label as u8 as char,
            });
        }

        // conv_idx assigns IDs in original A/B/C first-appearance order.
        // get_perm changes tensor roles but does not renormalize those IDs.
        let mut global_labels = Vec::new();
        for labels in indices {
            for label in labels.bytes() {
                if !global_labels.contains(&label) {
                    global_labels.push(label);
                }
            }
        }
        let group = |left: usize, right: usize| {
            labels_with_mask(
                &global_labels,
                &masks,
                (1 << roles[left]) | (1 << roles[right]),
            )
        };
        let role_k_labels = group(0, 1);
        let role_m_labels = group(0, 2);
        let role_n_labels = group(1, 2);
        let l_labels = labels_with_mask(&global_labels, &masks, 0b111);

        // lnmk is invariant and remains defined by the original A/B/C roles.
        let k_labels = labels_with_mask(&global_labels, &masks, 0b011);
        let m_labels = labels_with_mask(&global_labels, &masks, 0b101);
        let n_labels = labels_with_mask(&global_labels, &masks, 0b110);

        let k = label_product(&k_labels, &dimensions)?;
        let m = label_product(&m_labels, &dimensions)?;
        let n = label_product(&n_labels, &dimensions)?;
        let batches = label_product(&l_labels, &dimensions)?;
        let size_a = checked_product(shapes[0])?;
        let size_b = checked_product(shapes[1])?;
        let size_c = checked_product(shapes[2])?;
        let packed_size_a = k
            .checked_mul(m)
            .and_then(|size| size.checked_mul(batches))
            .ok_or(Rejected::SizeOverflow)?;
        let packed_size_b = k
            .checked_mul(n)
            .and_then(|size| size.checked_mul(batches))
            .ok_or(Rejected::SizeOverflow)?;
        let packed_size_c = m
            .checked_mul(n)
            .and_then(|size| size.checked_mul(batches))
            .ok_or(Rejected::SizeOverflow)?;
        debug_assert_eq!(size_a, packed_size_a);
        debug_assert_eq!(size_b, packed_size_b);
        debug_assert_eq!(size_c, packed_size_c);

        let mut packed_first = Vec::new();
        packed_first.extend_from_slice(&role_k_labels);
        packed_first.extend_from_slice(&role_m_labels);
        packed_first.extend_from_slice(&l_labels);
        let mut packed_second = Vec::new();
        packed_second.extend_from_slice(&role_k_labels);
        packed_second.extend_from_slice(&role_n_labels);
        packed_second.extend_from_slice(&l_labels);
        let mut packed_third = Vec::new();
        packed_third.extend_from_slice(&role_m_labels);
        packed_third.extend_from_slice(&role_n_labels);
        packed_third.extend_from_slice(&l_labels);
        let mut orders: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
        orders[roles[0]] = axis_order(indices[roles[0]].as_bytes(), &packed_first);
        orders[roles[1]] = axis_order(indices[roles[1]].as_bytes(), &packed_second);
        orders[roles[2]] = axis_order(indices[roles[2]].as_bytes(), &packed_third);
        let (trans_a, trans_b, transposed_output) = TRANSPOSES[permutation];

        Ok(Self {
            shape_a: shapes[0].to_vec(),
            shape_b: shapes[1].to_vec(),
            shape_c: shapes[2].to_vec(),
            order_a: std::mem::take(&mut orders[0]),
            order_b: std::mem::take(&mut orders[1]),
            order_c: std::mem::take(&mut orders[2]),
            m,
            n,
            k,
            batches,
            size_a,
            size_b,
            size_c,
            permutation,
            transpose_seconds: [0.; 3],
            trans_a,
            trans_b,
            transposed_output,
        })
    }

    /// Pack, execute the existing batched GEMM kernel, and restore C's layout.
    pub fn execute<T, K>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        alpha: T,
        beta: T,
    )
    where
        T: Clone + PartialEq,
        crate::algebra::Arithmetic<T>: crate::algebra::Semiring<Element = T>,
        K: GemmKernel<T>,
    {
        assert_eq!(a.len(), self.size_a);
        assert_eq!(b.len(), self.size_b);
        assert_eq!(c.len(), self.size_c);

        let packed_a = pack(a, &self.shape_a, &self.order_a);
        let packed_b = pack(b, &self.shape_b, &self.order_b);
        let mut packed_c = pack(c, &self.shape_c, &self.order_c);
        folded::<T, K>(
            Folded {
                m: self.m,
                n: self.n,
                k: self.k,
                batches: self.batches,
                trans_a: self.trans_a,
                trans_b: self.trans_b,
                transposed_output: self.transposed_output,
            },
            &packed_a,
            &packed_b,
            &mut packed_c,
            alpha,
            beta,
        );
        unpack(&packed_c, c, &self.shape_c, &self.order_c);
    }
}

fn labels_with_mask(source: &[u8], masks: &[u8; 256], mask: u8) -> Vec<u8> {
    source
        .iter()
        .copied()
        .filter(|&label| masks[label as usize] == mask)
        .collect()
}

fn label_product(labels: &[u8], dimensions: &[Option<usize>; 256]) -> Result<usize, Rejected> {
    labels.iter().try_fold(1_usize, |product, &label| {
        product
            .checked_mul(dimensions[label as usize].unwrap())
            .ok_or(Rejected::SizeOverflow)
    })
}

fn checked_product(shape: &[usize]) -> Result<usize, Rejected> {
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or(Rejected::SizeOverflow)
    })
}

fn axis_order(source_labels: &[u8], packed_labels: &[u8]) -> Vec<usize> {
    packed_labels
        .iter()
        .map(|label| {
            source_labels
                .iter()
                .position(|candidate| candidate == label)
                .unwrap()
        })
        .collect()
}

fn pack<T: Clone>(source: &[T], source_shape: &[usize], order: &[usize]) -> Vec<T> {
    let source_strides = strides(source_shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| source_shape[axis]).collect();
    let mut packed = source.to_vec();
    for (packed_offset, value) in packed.iter_mut().enumerate() {
        let mut remainder = packed_offset;
        let mut source_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            let coordinate = remainder % dimension;
            remainder /= dimension;
            source_offset += coordinate * source_strides[axis];
        }
        value.clone_from(&source[source_offset]);
    }
    packed
}

fn unpack<T: Clone>(packed: &[T], destination: &mut [T], source_shape: &[usize], order: &[usize]) {
    let source_strides = strides(source_shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| source_shape[axis]).collect();
    for (packed_offset, value) in packed.iter().enumerate() {
        let mut remainder = packed_offset;
        let mut destination_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            let coordinate = remainder % dimension;
            remainder /= dimension;
            destination_offset += coordinate * source_strides[axis];
        }
        destination[destination_offset].clone_from(value);
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
