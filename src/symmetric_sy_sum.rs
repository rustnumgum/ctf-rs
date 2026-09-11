// Full SY-aware summation orchestration adapted from cc4s CTF summation.cxx
// and symmetry/symmetrization.cxx. Copyright (c) 2011, Edgar Solomonik.
// See LICENSE.
use crate::{
    algebra::{Group, Semiring, Wire},
    scalar_conversion::CastFromF64,
    sym_indices::{align_pair, summation_factor},
    sym_permutations,
    symmetric_distribution::SymmetricDistribution,
    symmetry::Symmetry,
};

use super::SymmetricTensor;

#[derive(Clone, Copy)]
enum BrokenLink {
    Input(usize),
    Output(usize),
}

fn labels(indices: &[u8]) -> &str {
    // The public entry point establishes the ASCII invariant, and every
    // internal index map is only a permutation or identification of it.
    std::str::from_utf8(indices).unwrap()
}

fn assert_unique_indices(indices: &str, order: usize) {
    assert!(indices.is_ascii());
    assert_eq!(indices.len(), order);
    for (axis, label) in indices.bytes().enumerate() {
        assert!(
            !indices.as_bytes()[..axis].contains(&label),
            "sum_from internal symmetry unfolding requires unique labels"
        );
    }
}

fn contains_sy(links: &[Symmetry]) -> bool {
    links.contains(&Symmetry::SY)
}

fn axis_indices(order: usize) -> Vec<u8> {
    assert!(order <= 128, "symmetric index maps require at most 128 axes");
    (0..order).map(|axis| axis as u8).collect()
}

/// `summation::unfold_broken_sym`, including its additional fully reduced SY
/// case. The returned axis is the link replaced by NS in the unfolded tensor.
fn broken_link(
    input_indices: &[u8],
    input_links: &[Symmetry],
    output_indices: &[u8],
    output_links: &[Symmetry],
) -> Option<BrokenLink> {
    for i in 0..input_indices.len() {
        if input_links[i] == Symmetry::NS {
            continue;
        }
        let left = output_indices
            .iter()
            .rposition(|&label| label == input_indices[i]);
        let right = output_indices
            .iter()
            .rposition(|&label| label == input_indices[i + 1]);
        match left {
            Some(j)
                if output_links[j] == Symmetry::NS
                    || ((output_links[j] == Symmetry::AS)
                        ^ (input_links[i] == Symmetry::AS))
                    || right.is_none()
                    || output_indices[j + 1] != input_indices[i + 1] =>
            {
                return Some(BrokenLink::Input(i));
            }
            None if right.is_some() => return Some(BrokenLink::Input(i)),
            _ => {}
        }
    }

    for i in 0..output_indices.len() {
        if output_links[i] == Symmetry::NS {
            continue;
        }
        let left = input_indices
            .iter()
            .rposition(|&label| label == output_indices[i]);
        let right = input_indices
            .iter()
            .rposition(|&label| label == output_indices[i + 1]);
        match left {
            Some(j)
                if input_links[j] == Symmetry::NS
                    || ((input_links[j] == Symmetry::AS)
                        ^ (output_links[i] == Symmetry::AS))
                    || right.is_none()
                    || input_indices[j + 1] != output_indices[i + 1] =>
            {
                return Some(BrokenLink::Output(i));
            }
            None if right.is_some() => return Some(BrokenLink::Output(i)),
            _ => {}
        }
    }

    // A fully reduced symmetric pair is otherwise considered preserved by the
    // two link scans. CTF unfolds it so that coincidence surfaces occur once.
    for i in 0..input_indices.len() {
        if input_links[i] == Symmetry::SY
            && !output_indices.contains(&input_indices[i])
            && !output_indices.contains(&input_indices[i + 1])
        {
            return Some(BrokenLink::Input(i));
        }
    }

    None
}

impl<'c, 'r, A> SymmetricTensor<'c, 'r, A>
where
    A: Group + Semiring + Clone + CastFromF64,
    A::Element: Wire,
{
    /// Symmetry-aware indexed sum `B\[indices_b\] = alpha*A\[indices_a\]
    /// + beta*B\[indices_b\]` for NS/SY/AS/SH tensors.
    ///
    /// Repeated labels are extracted one pair at a time through this same
    /// symmetry-aware path with `run_diag` enabled, then output diagonals are
    /// restored in reverse order.
    pub fn sum_from(
        &mut self,
        indices_b: &str,
        a: &Self,
        indices_a: &str,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(std::ptr::eq(self.context, a.context));
        assert!(indices_a.is_ascii());
        assert!(indices_b.is_ascii());
        assert_eq!(indices_a.len(), a.distribution.links().len());
        assert_eq!(indices_b.len(), self.distribution.links().len());

        let repeated = [indices_a, indices_b].iter().any(|indices| {
            indices
                .bytes()
                .enumerate()
                .any(|(axis, label)| indices.as_bytes()[..axis].contains(&label))
        });
        if repeated {
            let (a, ia) = a.extract_diagonal(indices_a);
            let (mut b, ib) = self.extract_diagonal(indices_b);
            b.sum_from(&ib, &a, &ia, alpha, beta);
            self.replace_diagonal(indices_b, &b);
            return;
        }

        assert_unique_indices(indices_a, a.distribution.links().len());
        assert_unique_indices(indices_b, self.distribution.links().len());
        self.sum_sy_recursive(
            indices_b.as_bytes(),
            a,
            indices_a.as_bytes(),
            alpha,
            beta,
        );
    }

    /// Internal `sym_sum_tsr(run_diag=true)` entry used by one-axis diagonal
    /// extraction/reinsertion.  It bypasses only automatic diagonal
    /// preprocessing; symmetry alignment and unfolding remain active.
    pub(super) fn sum_from_run_diag(
        &mut self,
        output_indices: &[u8],
        input: &Self,
        input_indices: &[u8],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(std::ptr::eq(self.context, input.context));
        assert!(input_indices.is_ascii());
        assert!(output_indices.is_ascii());
        assert_eq!(input_indices.len(), input.distribution.links().len());
        assert_eq!(output_indices.len(), self.distribution.links().len());
        self.sum_sy_recursive(output_indices, input, input_indices, alpha, beta);
    }

    fn sum_sy_recursive(
        &mut self,
        output_indices: &[u8],
        input: &Self,
        input_indices: &[u8],
        alpha: A::Element,
        beta: A::Element,
    ) {
        let mut output_indices = output_indices.to_vec();
        let sign = align_pair(
            input_indices,
            input.distribution.links(),
            &mut output_indices,
            self.distribution.links(),
        );
        let factor = summation_factor(
            input_indices,
            input.distribution.links(),
            &output_indices,
        );
        let mut adjusted_alpha = self.algebra.zero();
        for _ in 0..factor {
            adjusted_alpha = self.algebra.add(&adjusted_alpha, &alpha);
        }
        if sign == -1 {
            adjusted_alpha = self.algebra.negate(&adjusted_alpha);
        }

        let Some(first) = broken_link(
            input_indices,
            input.distribution.links(),
            &output_indices,
            self.distribution.links(),
        ) else {
            self.sum_canonical_from(
                labels(&output_indices),
                input,
                labels(input_indices),
                adjusted_alpha,
                beta,
            );
            return;
        };

        let mut unfolded_input_links = input.distribution.links().to_vec();
        let mut unfolded_output_links = self.distribution.links().to_vec();
        match first {
            BrokenLink::Input(axis) => unfolded_input_links[axis] = Symmetry::NS,
            BrokenLink::Output(axis) => unfolded_output_links[axis] = Symmetry::NS,
        }

        let second = broken_link(
            input_indices,
            &unfolded_input_links,
            &output_indices,
            &unfolded_output_links,
        );
        let sy = match first {
            BrokenLink::Input(axis) => {
                input.distribution.links()[axis] == Symmetry::SY
                    || unfolded_input_links[axis] == Symmetry::SY
            }
            BrokenLink::Output(axis) => {
                self.distribution.links()[axis] == Symmetry::SY
                    || unfolded_output_links[axis] == Symmetry::SY
            }
        };

        // This is the source branch condition: a lone non-SY break is handled
        // by explicit permutations, whereas an input SY break always unfolds;
        // an output SY break unfolds only when beta is nonzero.
        if second.is_some()
            || (sy
                && (matches!(first, BrokenLink::Input(_)) || beta != self.algebra.zero()))
        {
            match first {
                BrokenLink::Input(_) => {
                    let relaxed_input = Self::desymmetrized(
                        input,
                        unfolded_input_links,
                        input_indices,
                        false,
                    );
                    self.sum_sy_recursive(
                        &output_indices,
                        &relaxed_input,
                        input_indices,
                        // `new_sum` is copied before this level installs its
                        // aligned/overcounted coefficient.
                        alpha,
                        beta,
                    );
                }
                BrokenLink::Output(_) => {
                    let mut relaxed_output = Self::desymmetrized(
                        self,
                        unfolded_output_links,
                        &output_indices,
                        true,
                    );
                    relaxed_output.sum_sy_recursive(
                        &output_indices,
                        input,
                        input_indices,
                        // Preserve the incoming coefficient, as in the copied
                        // source summation object.
                        alpha,
                        beta.clone(),
                    );

                    self.scale(&beta);
                    self.symmetrize_from(&relaxed_output, &output_indices);
                }
            }
            return;
        }

        let permutations = sym_permutations::enumerate::<2>(
            [input_indices, &output_indices],
            [input.distribution.links(), self.distribution.links()],
        );
        let mut task_beta = beta;
        for permutation in permutations {
            let task_alpha = if permutation.sign == 1 {
                adjusted_alpha.clone()
            } else {
                self.algebra.negate(&adjusted_alpha)
            };
            self.sum_canonical_from(
                labels(&permutation.indices[1]),
                input,
                labels(&permutation.indices[0]),
                task_alpha,
                task_beta,
            );
            task_beta = self.algebra.one();
        }
    }

    /// Active `desymmetrize` path. For SY, CTF emits one raw transpose task per
    /// other group axis plus the identity. The two-axis transpose excludes its
    /// diagonal, while larger groups retain the source's sequential
    /// coincidence-surface scaling (including its documented FIXME behavior).
    pub(super) fn desymmetrized(
        source: &Self,
        target_links: Vec<Symmetry>,
        _indices: &[u8],
        is_output: bool,
    ) -> Self {
        let distribution = SymmetricDistribution::new(
            source.distribution.distribution().clone(),
            target_links,
        );
        let mut result = Self::new(source.context, distribution, source.algebra.clone());

        let source_links = source.distribution.links();
        let target_links = result.distribution.links();
        let is = source_links
            .iter()
            .zip(target_links)
            .position(|(source, target)| source != target)
            .expect("desymmetrize requires one relaxed symmetry link");

        // Product desymmetrization intentionally creates a zero work tensor;
        // the recursive sum fills it before the old product is symmetrized in.
        if is_output {
            return result;
        }

        // Source desymmetrization always constructs identity index maps; the
        // outer operation's possibly repeated map is not propagated here.
        let indices = axis_indices(source_links.len());

        if source_links[is] != Symmetry::SY {
            if !contains_sy(source_links) && !contains_sy(target_links) {
                result.sum_hollow_from(
                    labels(&indices),
                    source,
                    labels(&indices),
                    source.algebra.one(),
                    source.algebra.one(),
                );
            } else {
                result.sum_sy_recursive(
                    &indices,
                    source,
                    &indices,
                    source.algebra.one(),
                    source.algebra.one(),
                );
            }
            return result;
        }

        let mut pivot = is;
        if is > 0 && source_links[is - 1] != Symmetry::NS {
            pivot += 1;
        }
        let mut positive = 0usize;
        while pivot + positive < source_links.len()
            && source_links[pivot + positive] != Symmetry::NS
        {
            positive += 1;
        }
        let mut negative = 0usize;
        while pivot > negative && source_links[pivot - negative - 1] != Symmetry::NS {
            negative += 1;
        }
        let other_axes = positive + negative;

        let diagonal_free;
        let transpose_source = if other_axes == 1 {
            let mut hollow_links = source_links.to_vec();
            hollow_links[is] = Symmetry::SH;
            diagonal_free = source.repack_groups(hollow_links);
            &diagonal_free
        } else {
            source
        };

        for relative in -(negative as isize) - 1..positive as isize {
            if relative == -1 {
                continue;
            }
            let partner = (pivot as isize + relative + 1) as usize;
            let mut transposed = indices.clone();
            transposed.swap(pivot, partner);
            result.sum_canonical_from(
                labels(&indices),
                transpose_source,
                labels(&transposed),
                source.algebra.one(),
                source.algebra.one(),
            );
        }
        result.sum_canonical_from(
            labels(&indices),
            source,
            labels(&indices),
            source.algebra.one(),
            source.algebra.one(),
        );

        if other_axes > 1 {
            let coincidence_scale = (other_axes - 1) as f64 / other_axes as f64;
            let coincidence_scale = result.algebra.cast_f64(coincidence_scale);
            for relative in -(negative as isize) - 1..positive as isize {
                if relative == -1 {
                    continue;
                }
                let partner = (pivot as isize + relative + 1) as usize;
                let mut diagonal = indices.clone();
                diagonal[partner] = diagonal[pivot];
                result.scale_indexed(labels(&diagonal), &coincidence_scale);
            }
        }

        result
    }

    /// Active source `symmetrize` path: sum into a fresh tensor with the target
    /// symmetry, then accumulate that tensor into the old target. The disabled
    /// fractional-rescaling implementation in the source is intentionally not
    /// reproduced.
    pub(super) fn symmetrize_from(&mut self, nonsymmetric: &Self, indices: &[u8]) {
        // As in CTF `symmetrize`, these transfers use identity maps rather
        // than the outer summation's (potentially repeated) map.
        let indices = axis_indices(indices.len());
        let mut intermediate = Self::new(
            self.context,
            self.distribution.clone(),
            self.algebra.clone(),
        );
        if !contains_sy(self.distribution.links())
            && !contains_sy(nonsymmetric.distribution.links())
        {
            intermediate.sum_hollow_from(
                labels(&indices),
                nonsymmetric,
                labels(&indices),
                self.algebra.one(),
                self.algebra.zero(),
            );
            self.sum_hollow_from(
                labels(&indices),
                &intermediate,
                labels(&indices),
                self.algebra.one(),
                self.algebra.one(),
            );
        } else {
            intermediate.sum_sy_recursive(
                &indices,
                nonsymmetric,
                &indices,
                self.algebra.one(),
                self.algebra.zero(),
            );
            self.sum_sy_recursive(
                &indices,
                &intermediate,
                &indices,
                self.algebra.one(),
                self.algebra.one(),
            );
        }
    }
}
