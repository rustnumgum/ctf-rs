// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Full symmetry-aware contraction orchestration above the mapped packed kernel.
//!
//! The control flow follows `contraction::sym_contract`,
//! `contraction::unfold_broken_sym`, and the terminal SY diagonal prescaling in
//! the pinned cc4s CTF source.
use crate::{
    algebra::{Group, Semiring, Wire},
    mapping::{Distribution, Mapping, Topology},
    scalar_conversion::CastFromF64,
    sym_indices::{align_triple, contraction_factor},
    sym_permutations,
    symmetric_distribution::SymmetricDistribution,
    symmetry::Symmetry,
};

use super::SymmetricTensor;

#[derive(Clone, Copy)]
enum BrokenLink {
    A(usize),
    B(usize),
    C(usize),
}

fn labels(indices: &[u8]) -> &str {
    std::str::from_utf8(indices).unwrap()
}

fn validate_indices(indices: &str, shape: &[usize]) {
    assert!(indices.is_ascii());
    assert_eq!(indices.len(), shape.len());
    for (axis, label) in indices.bytes().enumerate() {
        for previous in 0..axis {
            if indices.as_bytes()[previous] == label {
                assert_eq!(shape[previous], shape[axis]);
            }
        }
    }
}

fn position(indices: &[u8], label: u8) -> Option<usize> {
    indices.iter().position(|&candidate| candidate == label)
}

/// `contraction::unfold_broken_sym`. Exactly one returned link is relaxed to
/// NS before the next recursive contraction.
fn broken_link(
    a_indices: &[u8],
    a_links: &[Symmetry],
    b_indices: &[u8],
    b_links: &[Symmetry],
    c_indices: &[u8],
    c_links: &[Symmetry],
    same_inputs: bool,
) -> Option<BrokenLink> {
    for axis in 0..a_indices.len() {
        if a_links[axis] == Symmetry::NS {
            continue;
        }
        let left = a_indices[axis];
        let right = a_indices[axis + 1];
        if let Some(other) = position(b_indices, left) {
            if b_links[other] != a_links[axis] || b_indices[other + 1] != right {
                return Some(BrokenLink::A(axis));
            }
        } else if position(b_indices, right).is_some() {
            return Some(BrokenLink::A(axis));
        }
        if let Some(other) = position(c_indices, left) {
            if c_links[other] != a_links[axis] || c_indices[other + 1] != right {
                return Some(BrokenLink::A(axis));
            }
        } else if position(c_indices, right).is_some() {
            return Some(BrokenLink::A(axis));
        }
    }

    for axis in 0..b_indices.len() {
        if b_links[axis] == Symmetry::NS {
            continue;
        }
        let left = b_indices[axis];
        let right = b_indices[axis + 1];
        if let Some(other) = position(a_indices, left) {
            if a_links[other] != b_links[axis] || a_indices[other + 1] != right {
                return Some(BrokenLink::B(axis));
            }
        } else if position(a_indices, right).is_some() {
            return Some(BrokenLink::B(axis));
        }
        if let Some(other) = position(c_indices, left) {
            if c_links[other] != b_links[axis] || c_indices[other + 1] != right {
                return Some(BrokenLink::B(axis));
            }
        } else if position(c_indices, right).is_some() {
            return Some(BrokenLink::B(axis));
        }
    }

    // When A and B are the same tensor, the source preserves C symmetry if
    // every differently named input pair is joined by an all-SY run in C.
    let mut output_preserved = same_inputs;
    if output_preserved {
        for axis in 0..a_indices.len() {
            if a_indices[axis] == b_indices[axis] {
                continue;
            }
            let Some(a_output) = position(c_indices, a_indices[axis]) else {
                output_preserved = false;
                break;
            };
            let Some(b_output) = position(c_indices, b_indices[axis]) else {
                output_preserved = false;
                break;
            };
            if c_links[a_output.min(b_output)..a_output.max(b_output)]
                .iter()
                .any(|&link| link != Symmetry::SY)
            {
                output_preserved = false;
                break;
            }
        }
    }

    if !output_preserved {
        for axis in 0..c_indices.len() {
            if c_links[axis] == Symmetry::NS {
                continue;
            }
            let left = c_indices[axis];
            let right = c_indices[axis + 1];
            if let Some(other) = position(b_indices, left) {
                if b_links[other] != c_links[axis] || b_indices[other + 1] != right {
                    return Some(BrokenLink::C(axis));
                }
            } else if position(b_indices, right).is_some() {
                return Some(BrokenLink::C(axis));
            }
            if let Some(other) = position(a_indices, left) {
                if a_links[other] != c_links[axis] || a_indices[other + 1] != right {
                    return Some(BrokenLink::C(axis));
                }
            } else if position(a_indices, right).is_some() {
                return Some(BrokenLink::C(axis));
            }
        }
    }

    None
}

/// Test the explicit mapping at the same point where the source chooses
/// desymmetrization versus operation-permutation expansion.
fn mapping_supported(
    operands: [(&[u8], &[Symmetry], &[usize]); 3],
    topology: &Topology,
    physical_labels: &[u8],
) -> Result<[usize; 3], crate::map_tensor::Rejected> {
    let mut union = Vec::new();
    for (indices, _, _) in operands {
        for &label in indices {
            if !union.contains(&label) {
                union.push(label);
            }
        }
    }

    let mut maps = vec![Mapping::Unmapped; union.len()];
    for (topology_axis, &label) in physical_labels.iter().enumerate() {
        let axis = position(&union, label)
            .expect("each physical topology axis must name a union label");
        maps[axis].augment_physical(topology, topology_axis);
    }

    let mut table = vec![false; union.len() * union.len()];
    for (indices, links, _) in operands {
        for axis in 0..links.len().saturating_sub(1) {
            if links[axis] == Symmetry::NS {
                continue;
            }
            let left = position(&union, indices[axis]).unwrap();
            let right = position(&union, indices[axis + 1]).unwrap();
            table[left * union.len() + right] = true;
            table[right * union.len() + left] = true;
        }
    }
    crate::map_tensor::coordinate_symmetry(&mut maps, &table)?;

    Ok(std::array::from_fn(|operand| {
        let (indices, links, shape) = operands[operand];
        let mappings = indices
            .iter()
            .map(|&label| maps[position(&union, label).unwrap()].clone())
            .collect();
        SymmetricDistribution::new(
            Distribution::new(shape.to_vec(), topology.clone(), mappings),
            links.to_vec(),
        )
        .local_len()
    }))
}

/// Identify the SY runs selected by `prescale_operands`. The first operand is
/// the source's smaller T and the second is V.
fn prescale_masks(
    t_indices: &[u8],
    t_links: &[Symmetry],
    v_indices: &[u8],
    v_links: &[Symmetry],
    c_indices: &[u8],
) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
    let mut t_masks = Vec::new();
    let mut axis = 0;
    while axis < t_indices.len() {
        let initial_v = position(v_indices, t_indices[axis]);
        let mut current_v = initial_v;
        let mut current_c = position(c_indices, t_indices[axis]);
        let mut count = 0;
        while t_links[axis + count] == Symmetry::SY
            && current_c.is_none()
            && match (initial_v, current_v) {
                (None, None) => true,
                (Some(_), Some(v_axis)) => v_links[v_axis] == Symmetry::SY,
                _ => false,
            }
        {
            count += 1;
            current_v = position(v_indices, t_indices[axis + count]);
            current_c = position(c_indices, t_indices[axis + count]);
        }
        if t_links[axis + count] == Symmetry::NS
            && current_c.is_none()
            && match (initial_v, current_v) {
                (None, None) => true,
                (Some(_), Some(v_axis)) => v_links[v_axis] == Symmetry::NS,
                _ => false,
            }
        {
            count += 1;
        }
        if count > 1 {
            let mut mask = vec![false; t_indices.len()];
            mask[axis..axis + count].fill(true);
            t_masks.push(mask);
        }
        axis += count.max(1);
    }

    let mut v_masks = Vec::new();
    let mut axis = 0;
    while axis < v_indices.len() {
        let original_c = position(c_indices, v_indices[axis]);
        let mut current_t = position(t_indices, v_indices[axis]);
        let mut count = 0;
        while v_links[axis + count] == Symmetry::SY
            && original_c.is_none()
            && current_t.is_none()
        {
            count += 1;
            current_t = position(t_indices, v_indices[axis + count]);
        }
        if v_links[axis + count] == Symmetry::NS
            && original_c.is_none()
            && current_t.is_none()
        {
            count += 1;
        }
        if count > 1 {
            let mut mask = vec![false; v_indices.len()];
            mask[axis..axis + count].fill(true);
            v_masks.push(mask);
        }
        axis += count.max(1);
    }
    (t_masks, v_masks)
}

impl<'c, 'r, A> SymmetricTensor<'c, 'r, A>
where
    A: Group + Semiring + Clone + CastFromF64,
    A::Element: Wire,
{
    fn cloned(&self) -> Self {
        Self {
            context: self.context,
            distribution: self.distribution.clone(),
            algebra: self.algebra.clone(),
            data: self.data.clone(),
        }
    }

    /// `scale_diagonals`: divide a canonical value by the number of marked-axis
    /// permutations that leave its coordinate unchanged.
    fn scale_diagonals(&mut self, mask: &[bool]) {
        let algebra = self.algebra.clone();
        let distribution = self.distribution.distribution().clone();
        for (offset, key) in self.distribution.local_pairs(self.context.rank()) {
            let coordinates = distribution.decode_key(key);
            let mut factor = 1usize;
            for axis in 0..mask.len() {
                if !mask[axis] {
                    continue;
                }
                let equal = (axis + 1..mask.len())
                    .filter(|&other| mask[other] && coordinates[other] == coordinates[axis])
                    .count();
                factor *= equal + 1;
            }
            if factor != 1 {
                let reciprocal = algebra.cast_f64(1.0 / factor as f64);
                self.data[offset] = algebra.multiply(&self.data[offset], &reciprocal);
            }
        }
    }

    /// Execute one symmetry-disabled packed task, including the source's
    /// fully-reduced-SY coincidence prescaling of input operands.
    fn contract_raw_on(
        &mut self,
        c_indices: &[u8],
        a: &Self,
        a_indices: &[u8],
        b: &Self,
        b_indices: &[u8],
        topology: &Topology,
        physical_labels: &[u8],
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) -> Result<(), crate::map_tensor::Rejected> {
        let mapped_sizes = mapping_supported(
            [
                (
                    a_indices,
                    a.distribution.links(),
                    &a.distribution.distribution().shape,
                ),
                (
                    b_indices,
                    b.distribution.links(),
                    &b.distribution.distribution().shape,
                ),
                (
                    c_indices,
                    self.distribution.links(),
                    &self.distribution.distribution().shape,
                ),
            ],
            topology,
            physical_labels,
        )?;
        let a_size = mapped_sizes[0];
        let b_size = mapped_sizes[1];
        let (t_indices, t_links, v_indices, v_links, t_is_a) = if a_size <= b_size {
            (
                a_indices,
                a.distribution.links(),
                b_indices,
                b.distribution.links(),
                true,
            )
        } else {
            (
                b_indices,
                b.distribution.links(),
                a_indices,
                a.distribution.links(),
                false,
            )
        };
        let (t_masks, v_masks) =
            prescale_masks(t_indices, t_links, v_indices, v_links, c_indices);

        if t_masks.is_empty() && v_masks.is_empty() {
            return self.contract_canonical_on(
                labels(c_indices),
                a,
                labels(a_indices),
                b,
                labels(b_indices),
                topology.clone(),
                labels(physical_labels),
                alpha,
                beta,
                commutative,
            );
        }

        let mut scaled_a = a.cloned();
        let mut scaled_b = b.cloned();
        if t_is_a {
            for mask in &t_masks {
                scaled_a.scale_diagonals(mask);
            }
            for mask in &v_masks {
                scaled_b.scale_diagonals(mask);
            }
        } else {
            for mask in &t_masks {
                scaled_b.scale_diagonals(mask);
            }
            for mask in &v_masks {
                scaled_a.scale_diagonals(mask);
            }
        }
        self.contract_canonical_on(
            labels(c_indices),
            &scaled_a,
            labels(a_indices),
            &scaled_b,
            labels(b_indices),
            topology.clone(),
            labels(physical_labels),
            alpha,
            beta,
            commutative,
        )
    }

    /// Symmetry-aware contraction
    /// `C[indices_c] = alpha*A[indices_a]*B[indices_b] + beta*C[indices_c]`
    /// on an explicit label-to-grid mapping.
    /// Repeated labels use the source run_diag extraction/reinsertion path.
    /// This entry point does not choose a topology or physical-label map.
    pub fn contract_from_on(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        topology: Topology,
        physical_labels: &str,
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) -> Result<(), crate::map_tensor::Rejected> {
        assert!(std::ptr::eq(self.context, a.context));
        assert!(std::ptr::eq(self.context, b.context));
        assert_eq!(topology.size(), self.context.size());
        assert!(physical_labels.is_ascii());
        assert_eq!(physical_labels.len(), topology.dimensions.len());
        validate_indices(indices_a, &a.distribution.distribution().shape);
        validate_indices(indices_b, &b.distribution.distribution().shape);
        validate_indices(indices_c, &self.distribution.distribution().shape);

        let operands = [
            (indices_a, &a.distribution.distribution().shape),
            (indices_b, &b.distribution.distribution().shape),
            (indices_c, &self.distribution.distribution().shape),
        ];
        let mut dimensions: Vec<(u8, usize)> = Vec::new();
        for (indices, shape) in operands {
            for (axis, label) in indices.bytes().enumerate() {
                if let Some((_, dimension)) =
                    dimensions.iter().find(|(candidate, _)| *candidate == label)
                {
                    assert_eq!(*dimension, shape[axis]);
                } else {
                    dimensions.push((label, shape[axis]));
                }
            }
        }
        for label in physical_labels.bytes() {
            assert!(dimensions.iter().any(|(candidate, _)| *candidate == label));
        }

        let repeated = [indices_a, indices_b, indices_c].iter().any(|indices| {
            indices
                .bytes()
                .enumerate()
                .any(|(axis, label)| indices.as_bytes()[..axis].contains(&label))
        });
        if repeated {
            let original_indices_c = indices_c;
            let (a, indices_a) = a.extract_diagonal(indices_a);
            let (b, indices_b) = b.extract_diagonal(indices_b);
            let (mut c, indices_c) = self.extract_diagonal(indices_c);
            c.contract_from_on(
                &indices_c,
                &a,
                &indices_a,
                &b,
                &indices_b,
                topology,
                physical_labels,
                alpha,
                beta,
                commutative,
            )?;
            self.replace_diagonal(original_indices_c, &c);
            return Ok(());
        }

        self.contract_sy_recursive(
            indices_c.as_bytes(),
            a,
            indices_a.as_bytes(),
            b,
            indices_b.as_bytes(),
            &topology,
            physical_labels.as_bytes(),
            alpha,
            beta,
            commutative,
        )
    }

    fn contract_sy_recursive(
        &mut self,
        c_indices: &[u8],
        a: &Self,
        a_indices: &[u8],
        b: &Self,
        b_indices: &[u8],
        topology: &Topology,
        physical_labels: &[u8],
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) -> Result<(), crate::map_tensor::Rejected> {
        let mut b_indices = b_indices.to_vec();
        let mut c_indices = c_indices.to_vec();
        let sign = align_triple(
            a_indices,
            a.distribution.links(),
            &mut b_indices,
            b.distribution.links(),
            &mut c_indices,
            self.distribution.links(),
        );
        let factor = contraction_factor(
            a_indices,
            a.distribution.links(),
            &b_indices,
            b.distribution.links(),
            &c_indices,
        );
        let align_alpha = if sign == 1 {
            alpha.clone()
        } else {
            self.algebra.negate(&alpha)
        };
        let mut adjusted_alpha = self.algebra.zero();
        for _ in 0..factor {
            adjusted_alpha = self.algebra.add(&adjusted_alpha, &align_alpha);
        }

        let Some(broken) = broken_link(
            a_indices,
            a.distribution.links(),
            &b_indices,
            b.distribution.links(),
            &c_indices,
            self.distribution.links(),
            std::ptr::eq(a, b),
        ) else {
            return self.contract_raw_on(
                &c_indices,
                a,
                a_indices,
                b,
                &b_indices,
                topology,
                physical_labels,
                adjusted_alpha,
                beta,
                commutative,
            );
        };

        let mut a_links = a.distribution.links().to_vec();
        let mut b_links = b.distribution.links().to_vec();
        let mut c_links = self.distribution.links().to_vec();
        match broken {
            BrokenLink::A(axis) => a_links[axis] = Symmetry::NS,
            BrokenLink::B(axis) => b_links[axis] = Symmetry::NS,
            BrokenLink::C(axis) => c_links[axis] = Symmetry::NS,
        }

        if mapping_supported(
            [
                (a_indices, &a_links, &a.distribution.distribution().shape),
                (&b_indices, &b_links, &b.distribution.distribution().shape),
                (
                    &c_indices,
                    &c_links,
                    &self.distribution.distribution().shape,
                ),
            ],
            topology,
            physical_labels,
        )
        .is_ok()
        {
            match broken {
                BrokenLink::A(_) => {
                    let relaxed = Self::desymmetrized(a, a_links, a_indices, false);
                    self.contract_sy_recursive(
                        &c_indices,
                        &relaxed,
                        a_indices,
                        b,
                        &b_indices,
                        topology,
                        physical_labels,
                        align_alpha,
                        beta,
                        commutative,
                    )?;
                }
                BrokenLink::B(_) => {
                    let relaxed = Self::desymmetrized(b, b_links, &b_indices, false);
                    self.contract_sy_recursive(
                        &c_indices,
                        a,
                        a_indices,
                        &relaxed,
                        &b_indices,
                        topology,
                        physical_labels,
                        align_alpha,
                        beta,
                        commutative,
                    )?;
                }
                BrokenLink::C(_) => {
                    let mut relaxed = Self::desymmetrized(self, c_links, &c_indices, true);
                    relaxed.contract_sy_recursive(
                        &c_indices,
                        a,
                        a_indices,
                        b,
                        &b_indices,
                        topology,
                        physical_labels,
                        align_alpha,
                        beta.clone(),
                        commutative,
                    )?;
                    self.scale(&beta);
                    self.symmetrize_from(&relaxed, &c_indices);
                }
            }
            return Ok(());
        }

        let permutations = sym_permutations::enumerate::<3>(
            [a_indices, &b_indices, &c_indices],
            [
                a.distribution.links(),
                b.distribution.links(),
                self.distribution.links(),
            ],
        );
        let mut task_beta = beta;
        for permutation in permutations {
            let task_alpha = if permutation.sign == 1 {
                adjusted_alpha.clone()
            } else {
                self.algebra.negate(&adjusted_alpha)
            };
            self.contract_raw_on(
                &permutation.indices[2],
                a,
                &permutation.indices[0],
                b,
                &permutation.indices[1],
                topology,
                physical_labels,
                task_alpha,
                task_beta,
                commutative,
            )?;
            task_beta = self.algebra.one();
        }
        Ok(())
    }
}
