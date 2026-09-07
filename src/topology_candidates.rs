// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted from src/mapping/topology.cxx at the pinned upstream commit.
//! Topology candidate order is preserved because it participates in plan selection.
use crate::mapping::Topology;

/// Enumerate ordered factorizations using upstream's mixed-radix prime-power
/// divisor order (not sorted divisor order).
pub fn all_shapes(size: usize) -> Vec<Topology> {
    assert!(size > 0);
    let mut remaining = size;
    let mut primes = Vec::new();
    let mut multiplicities = Vec::new();
    let mut divisor = 2;
    while remaining > 1 {
        if remaining % divisor == 0 {
            let mut count = 0;
            while remaining % divisor == 0 {
                remaining /= divisor;
                count += 1;
            }
            primes.push(divisor);
            multiplicities.push(count);
        }
        divisor += 1;
    }
    fn enumerate(
        primes: &[usize],
        mults: &[usize],
        prefix: &mut Vec<usize>,
        result: &mut Vec<Topology>,
    ) {
        let divisors: usize = mults.iter().map(|m| m + 1).product();
        assert!(divisors < 1_000_000);
        if divisors == 1 {
            result.push(Topology::new(prefix.clone()));
            return;
        }
        for divisor in 1..divisors {
            let mut quotient = divisor;
            let mut length = 1;
            let mut rest = Vec::with_capacity(mults.len());
            for (&prime, &count) in primes.iter().zip(mults) {
                let exponent = quotient % (count + 1);
                quotient /= count + 1;
                length *= prime.pow(exponent as u32);
                rest.push(count - exponent);
            }
            prefix.push(length);
            enumerate(primes, &rest, prefix, result);
            prefix.pop();
        }
    }
    let mut result = Vec::new();
    enumerate(&primes, &multiplicities, &mut Vec::new(), &mut result);
    result
}

/// Fold adjacent topology dimensions, retaining upstream's discovery order.
pub fn peel(topology: &Topology) -> Vec<Topology> {
    let mut result = vec![topology.clone()];
    let n = topology.dimensions.len();
    if n <= 1 {
        return result;
    }
    for i in 0..n - 1 {
        let mut dimensions = topology.dimensions.clone();
        dimensions[i] *= dimensions.remove(i + 1);
        result.push(Topology::new(dimensions));
    }
    let mut i = 1;
    while i < result.len() {
        for candidate in peel(&result[i]) {
            if !result.contains(&candidate) {
                result.push(candidate);
            }
        }
        i += 1;
    }
    result
}

/// Upstream peel_perm_torus: enumerate dimension swaps before folding.
pub fn peel_permutations(topology: &Topology) -> Vec<Topology> {
    let mut permutations = vec![topology.clone()];
    let mut i = 0;
    while i < permutations.len() {
        let dimensions = permutations[i].dimensions.clone();
        for j in 0..dimensions.len() {
            if dimensions[j] == 2 {
                continue;
            }
            for k in 0..dimensions.len() {
                if j == k || dimensions[j] == dimensions[k] {
                    continue;
                }
                let mut swapped = dimensions.clone();
                swapped.swap(j, k);
                let next = Topology::new(swapped);
                if !permutations.contains(&next) {
                    permutations.push(next);
                }
            }
        }
        i += 1;
    }
    let mut result = Vec::new();
    for permutation in permutations {
        for candidate in peel(&permutation) {
            if !result.contains(&candidate) {
                result.push(candidate);
            }
        }
    }
    result
}
