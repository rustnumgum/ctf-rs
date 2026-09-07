// Adapted from cc4s CTF interface/multilinear.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Auxiliary-index blocking selected from an explicit local memory budget.

use crate::{
    algebra::Monoid,
    context::Context,
    mapping::Distribution,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TttpBlocking {
    Divisions(usize),
    /// Per-rank available bytes used in the source working-set estimate.
    /// No OS-memory discovery or total-process peak-memory guarantee.
    AvailableBytes(u64),
}

#[derive(Clone)]
struct Maximum;

impl Monoid for Maximum {
    type Element = u64;

    fn zero(&self) -> u64 {
        0
    }

    fn add(&self, left: &u64, right: &u64) -> u64 {
        std::cmp::max(*left, *right)
    }
}

fn required_elements(
    mode_lengths_and_phases: &[(usize, usize)],
    auxiliary_length: usize,
    divisions: usize,
    local_pairs: usize,
) -> u64 {
    let block = auxiliary_length.div_ceil(divisions);
    let mut total = 0u64;
    for &(length, phase) in mode_lengths_and_phases {
        total += (2 * length * block / phase) as u64;
    }
    if divisions > 1 {
        total += local_pairs as u64;
    }
    total
}

fn local_divisions(
    mode_lengths_and_phases: &[(usize, usize)],
    auxiliary_length: usize,
    local_pairs: usize,
    element_bytes: usize,
    available_bytes: u64,
) -> usize {
    let mut divisions = 1;
    loop {
        let bytes = required_elements(
            mode_lengths_and_phases,
            auxiliary_length,
            divisions,
            local_pairs,
        ) * element_bytes as u64;
        if bytes <= available_bytes {
            return divisions;
        }
        assert_ne!(
            divisions, auxiliary_length,
            "insufficient memory for TTTP"
        );
        divisions = (2 * divisions).min(auxiliary_length);
    }
}

pub(crate) fn resolve(
    context: &Context<'_>,
    distribution: &Distribution,
    modes: &[usize],
    auxiliary_length: usize,
    local_pairs: usize,
    element_bytes: usize,
    blocking: TttpBlocking,
) -> usize {
    assert!(auxiliary_length > 0);
    match blocking {
        TttpBlocking::Divisions(divisions) => {
            assert!(divisions > 0 && divisions <= auxiliary_length);
            divisions
        }
        TttpBlocking::AvailableBytes(available_bytes) => {
            let facts: Vec<_> = modes
                .iter()
                .map(|&mode| {
                    (
                        distribution.shape[mode],
                        distribution.mappings[mode].physical_phase(),
                    )
                })
                .collect();
            let local = local_divisions(
                &facts,
                auxiliary_length,
                local_pairs,
                element_bytes,
                available_bytes,
            );
            let mut global = [local as u64];
            context.all_reduce_monoid(&Maximum, &mut global, true);
            global[0] as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{local_divisions, required_elements};

    #[test]
    fn source_sums_each_truncated_mode_term() {
        assert_eq!(required_elements(&[(3, 4), (3, 4)], 1, 1, 99), 2);
    }

    #[test]
    fn source_doubles_and_caps_divisions() {
        assert_eq!(local_divisions(&[(8, 1)], 10, 0, 1, 64), 4);
        assert_eq!(local_divisions(&[(8, 1)], 9, 0, 1, 32), 8);
        assert_eq!(local_divisions(&[(8, 1)], 9, 0, 1, 16), 9);
    }

    #[test]
    fn accumulator_is_counted_only_after_first_division() {
        assert_eq!(required_elements(&[(3, 1)], 4, 1, 17), 24);
        assert_eq!(required_elements(&[(3, 1)], 4, 2, 17), 29);
    }

    #[test]
    fn equality_fits_without_another_division() {
        assert_eq!(local_divisions(&[(5, 1)], 3, 0, 8, 240), 1);
    }

    #[test]
    #[should_panic(expected = "insufficient memory for TTTP")]
    fn source_fails_when_one_column_still_exceeds_budget() {
        local_divisions(&[(5, 1)], 3, 20, 8, 1);
    }
}
