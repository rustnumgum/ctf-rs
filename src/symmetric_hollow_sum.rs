// Symmetry-aware summation orchestration adapted from cc4s CTF summation.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{
    algebra::{Group, Semiring, Wire},
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

fn assert_unique_indices(indices: &str, order: usize) {
    assert!(indices.is_ascii());
    assert_eq!(indices.len(), order);
    for (i, label) in indices.bytes().enumerate() {
        assert!(
            !indices.as_bytes()[..i].contains(&label),
            "sum_hollow_from does not support repeated labels"
        );
    }
}

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
            .position(|&label| label == input_indices[i]);
        let right = output_indices
            .iter()
            .position(|&label| label == input_indices[i + 1]);
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
            .position(|&label| label == output_indices[i]);
        let right = input_indices
            .iter()
            .position(|&label| label == output_indices[i + 1]);
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

    None
}

fn labels(indices: &[u8]) -> &str {
    // The public entry point establishes the ASCII invariant, and alignment
    // and permutation only rearrange those same bytes.
    std::str::from_utf8(indices).unwrap()
}

impl<'c, 'r, A> SymmetricTensor<'c, 'r, A>
where
    A: Group + Semiring + Clone,
    A::Element: Wire,
{
    /// Symmetry-aware indexed sum for the hollow AS/SH domain.
    ///
    /// NS/AS/SH links, with repeated labels handled by the supported diagonal
    /// extraction primitive. Cross-group symmetry-breaking diagonals and SY
    /// coincidence-surface unfolding are not approximated here.
    pub fn sum_hollow_from(
        &mut self,
        output_indices: &str,
        input: &Self,
        input_indices: &str,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(std::ptr::eq(self.context, input.context));
        assert!(
            input
                .distribution
                .links()
                .iter()
                .chain(self.distribution.links())
                .all(|&link| link != Symmetry::SY),
            "sum_hollow_from does not support SY links"
        );

        let repeated = [input_indices, output_indices].iter().any(|labels|
            labels.bytes().enumerate().any(|(axis,label)| labels.as_bytes()[..axis].contains(&label)));
        if repeated {
            let (a, ia) = input.extract_diagonal(input_indices);
            let (mut b, ib) = self.extract_diagonal(output_indices);
            b.sum_hollow_from(&ib, &a, &ia, alpha, beta);
            self.replace_diagonal(output_indices, &b);
            return;
        }
        assert_unique_indices(input_indices, input.distribution.links().len());
        assert_unique_indices(output_indices, self.distribution.links().len());

        self.sum_hollow_recursive(
            output_indices.as_bytes(),
            input,
            input_indices.as_bytes(),
            alpha,
            beta,
        );
    }

    fn sum_hollow_recursive(
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

        // CTF forms integer factors through repeated addition so this remains
        // valid for an arbitrary scalar algebra. A zero factor cancels alpha.
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

        let mut relaxed_input_links = input.distribution.links().to_vec();
        let mut relaxed_output_links = self.distribution.links().to_vec();
        match first {
            BrokenLink::Input(axis) => relaxed_input_links[axis] = Symmetry::NS,
            BrokenLink::Output(axis) => relaxed_output_links[axis] = Symmetry::NS,
        }

        if broken_link(
            input_indices,
            &relaxed_input_links,
            &output_indices,
            &relaxed_output_links,
        )
        .is_none()
        {
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
            return;
        }

        match first {
            BrokenLink::Input(_) => {
                let distribution = SymmetricDistribution::new(
                    input.distribution.distribution().clone(),
                    relaxed_input_links,
                );
                let mut relaxed_input =
                    Self::new(input.context, distribution, input.algebra.clone());
                relaxed_input.sum_hollow_from(
                    labels(input_indices),
                    input,
                    labels(input_indices),
                    self.algebra.one(),
                    self.algebra.one(),
                );
                self.sum_hollow_recursive(
                    &output_indices,
                    &relaxed_input,
                    input_indices,
                    // new_sum was copied before this stage's adjusted-alpha
                    // pointer was installed, so recursive unfolding starts
                    // again from the incoming coefficient.
                    alpha,
                    beta,
                );
            }
            BrokenLink::Output(_) => {
                let distribution = SymmetricDistribution::new(
                    self.distribution.distribution().clone(),
                    relaxed_output_links,
                );
                let mut relaxed_output =
                    Self::new(self.context, distribution, self.algebra.clone());
                relaxed_output.sum_hollow_recursive(
                    &output_indices,
                    input,
                    input_indices,
                    // Match unfold_sum's copy of the pre-adjustment new_sum.
                    alpha,
                    beta.clone(),
                );

                self.scale(&beta);
                self.sum_hollow_from(
                    labels(&output_indices),
                    &relaxed_output,
                    labels(&output_indices),
                    self.algebra.one(),
                    self.algebra.one(),
                );
            }
        }
    }
}
