//! Sparse storage over symmetry-canonical keys.
//!
//! Values are stored in a [`SparseTensor`] using the rectangular distribution
//! carried by [`SymmetricDistribution`].  Only canonical keys are inserted;
//! indexed access applies the symmetry sign before reaching that storage.

use crate::{
    algebra::{Group, Semiring, Wire},
    context::Context,
    mapping::Distribution,
    sparse::SparseTensor,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry,
};

#[derive(Clone)]
pub struct SparseSymmetricTensor<'c, 'r, A: Group> {
    distribution: SymmetricDistribution,
    storage: SparseTensor<'c, 'r, A>,
}

impl<'c, 'r, A: Group> SparseSymmetricTensor<'c, 'r, A> {
    pub fn new(
        context: &'c Context<'r>,
        distribution: SymmetricDistribution,
        algebra: A,
    ) -> Self {
        let storage = SparseTensor::new(
            context,
            distribution.distribution().clone(),
            algebra,
        );
        Self {
            distribution,
            storage,
        }
    }

    pub fn context(&self) -> &'c Context<'r> {
        self.storage.context()
    }

    pub fn distribution(&self) -> &SymmetricDistribution {
        &self.distribution
    }

    pub fn algebra(&self) -> &A {
        self.storage.algebra()
    }

    /// Rank-local primary canonical entries. Backing replicas are omitted.
    pub fn local_pairs(&self) -> Vec<(usize, A::Element)> {
        let rank = self.context().rank();
        self.storage
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| self.distribution.distribution().owner(*key) == rank)
            .collect()
    }

    /// Number of rank-local primary canonical entries.
    pub fn local_nnz(&self) -> usize {
        self.local_pairs().len()
    }

    /// Number of logical entries represented by a canonical stored entry.
    pub fn orbit_multiplicity(&self, key: usize) -> usize {
        let Some((canonical, _)) = self.distribution.canonicalize(key) else {
            return 0;
        };
        let coordinates = self.distribution.distribution().decode_key(canonical);
        let mut multiplicity = 1;
        let mut start = 0;
        while start < self.distribution.links().len() {
            let mut end = start;
            while self.distribution.links()[end] != Symmetry::NS {
                end += 1;
            }
            for factor in 2..=end - start + 1 {
                multiplicity *= factor;
            }
            let mut run = start;
            while run <= end {
                let mut next = run + 1;
                while next <= end && coordinates[next] == coordinates[run] {
                    next += 1;
                }
                for factor in 2..=next - run {
                    multiplicity /= factor;
                }
                run = next;
            }
            start = end + 1;
        }
        multiplicity
    }
}

impl<'c, 'r, A: Group + Clone> SparseSymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Create sparse canonical storage from the packed dense representation.
    pub fn from_dense(
        source: &SymmetricTensor<'c, 'r, A>,
        mut keep: impl FnMut(&A::Element) -> bool,
    ) -> Self {
        let mut result = Self::new(
            source.context(),
            source.distribution().clone(),
            source.algebra().clone(),
        );
        let rank = source.context().rank();
        let pairs: Vec<_> = source
            .local_pairs()
            .into_iter()
            .filter(|(key, value)| {
                source.distribution().distribution().owner(*key) == rank && keep(value)
            })
            .collect();
        result.storage.write_add(&pairs);
        result
    }

    /// Collective indexed reads; AS permutations are negated and AS/SH
    /// structural diagonals read as zero.
    pub fn read(&self, keys: &[usize]) -> Vec<A::Element> {
        let mut canonical = Vec::new();
        let mut positions = Vec::new();
        for (position, &key) in keys.iter().enumerate() {
            if let Some((key, sign)) = self.distribution.canonicalize(key) {
                canonical.push(key);
                positions.push((position, sign));
            }
        }
        let values = self.storage.read(&canonical);
        let mut result = vec![self.algebra().zero(); keys.len()];
        for (value, (position, sign)) in values.into_iter().zip(positions) {
            result[position] = if sign == 1 {
                value
            } else {
                self.algebra().negate(&value)
            };
        }
        result
    }

    /// Collective additive writes after canonicalization. Equivalent AS
    /// permutations contribute with their canonical sign.
    pub fn write_add(&mut self, pairs: &[(usize, A::Element)]) {
        let pairs: Vec<_> = pairs
            .iter()
            .filter_map(|(key, value)| {
                self.distribution.canonicalize(*key).map(|(key, sign)| {
                    (
                        key,
                        if sign == 1 {
                            value.clone()
                        } else {
                            self.algebra().negate(value)
                        },
                    )
                })
            })
            .collect();
        self.storage.write_add(&pairs);
    }

    /// Redistribute stored canonical pairs without unpacking symmetry or
    /// allocating rectangular dense storage.
    pub fn redistribute(&mut self, target: SymmetricDistribution) {
        assert_eq!(target.links(), self.distribution.links());
        assert_eq!(
            target.distribution().shape,
            self.distribution.distribution().shape
        );
        self.storage.redistribute(target.distribution().clone());
        self.distribution = target;
    }

    /// Convert to packed dense symmetric storage. Only canonical source owners
    /// participate so replicated mappings do not duplicate values.
    pub fn into_dense(self) -> SymmetricTensor<'c, 'r, A> {
        let context = self.context();
        let algebra = self.algebra().clone();
        let rank = context.rank();
        let pairs: Vec<_> = self
            .storage
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| self.distribution.distribution().owner(*key) == rank)
            .collect();
        let mut result = SymmetricTensor::new(
            context,
            self.distribution,
            algebra,
        );
        result.write_add(&pairs);
        result
    }

    /// Expand canonical entries to a conventional sparse tensor. The result
    /// remains sparse; no rectangular dense allocation is used.
    pub fn unpack_orbits(&self, target: Distribution) -> SparseTensor<'c, 'r, A> {
        assert_eq!(target.shape, self.distribution.distribution().shape);
        let mut result = SparseTensor::new(self.context(), target, self.algebra().clone());
        result.write_add(&self.local_orbit_pairs());
        result
    }

    pub(crate) fn local_orbit_pairs(&self) -> Vec<(usize, A::Element)> {
        self.local_pairs()
            .into_iter()
            .flat_map(|(key, value)| {
                orbit(&self.distribution, key)
                    .into_iter()
                    .map(move |(key, sign)| {
                        let value = if sign == 1 {
                            value.clone()
                        } else {
                            self.algebra().negate(&value)
                        };
                        (key, value)
                    })
            })
            .collect()
    }
}

impl<A: Group + Semiring + Clone> SparseSymmetricTensor<'_, '_, A>
where
    A::Element: Wire,
{
    /// Same-index sparse symmetric summation used by source expressions such
    /// as `F2["ij"] += F["ij"]`.
    pub fn sum_from(
        &mut self,
        indices_b: &str,
        source: &Self,
        indices_a: &str,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(std::ptr::eq(self.context(), source.context()));
        assert_eq!(indices_b, indices_a);
        assert_eq!(self.distribution.links(), source.distribution.links());
        assert_eq!(
            self.distribution.distribution().shape,
            source.distribution.distribution().shape
        );
        self.storage.scale(&beta);
        let rank = source.context().rank();
        let pairs: Vec<_> = source
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| source.distribution.distribution().owner(*key) == rank)
            .map(|(key, value)| (key, self.algebra().multiply(&value, &alpha)))
            .collect();
        self.storage.write_add(&pairs);
    }

    pub fn add_inverse(&mut self) {
        let algebra = self.algebra().clone();
        self.storage
            .transform_stored(|_, value| *value = algebra.negate(value));
    }
}

fn orbit(distribution: &SymmetricDistribution, key: usize) -> Vec<(usize, i32)> {
    let coordinates = distribution.distribution().decode_key(key);
    let mut result = vec![(coordinates, 1)];
    let mut start = 0;
    while start < distribution.links().len() {
        let mut end = start;
        while distribution.links()[end] != Symmetry::NS {
            end += 1;
        }
        if end > start {
            let kind = distribution.links()[start];
            let values = result[0].0[start..=end].to_vec();
            let permutations = permutations(&values, kind == Symmetry::AS);
            let mut expanded = Vec::with_capacity(result.len() * permutations.len());
            for (coordinates, outer_sign) in result {
                for (permutation, inner_sign) in &permutations {
                    let mut coordinates = coordinates.clone();
                    coordinates[start..=end].clone_from_slice(permutation);
                    expanded.push((coordinates, outer_sign * inner_sign));
                }
            }
            result = expanded;
        }
        start = end + 1;
    }
    result
        .into_iter()
        .map(|(coordinates, sign)| (distribution.distribution().encode_key(&coordinates), sign))
        .collect()
}

fn permutations(values: &[usize], antisymmetric: bool) -> Vec<(Vec<usize>, i32)> {
    fn visit(
        values: &[usize],
        used: &mut [bool],
        order: &mut Vec<usize>,
        result: &mut Vec<(Vec<usize>, i32)>,
        antisymmetric: bool,
    ) {
        if order.len() == values.len() {
            let inversions = order
                .iter()
                .enumerate()
                .map(|(i, &left)| order[i + 1..].iter().filter(|&&right| left > right).count())
                .sum::<usize>();
            result.push((
                order.iter().map(|&position| values[position]).collect(),
                if antisymmetric && inversions % 2 == 1 {
                    -1
                } else {
                    1
                },
            ));
            return;
        }
        for position in 0..values.len() {
            if used[position]
                || (0..position).any(|previous| {
                    !used[previous] && values[previous] == values[position]
                })
            {
                continue;
            }
            used[position] = true;
            order.push(position);
            visit(values, used, order, result, antisymmetric);
            order.pop();
            used[position] = false;
        }
    }

    let mut result = Vec::new();
    visit(
        values,
        &mut vec![false; values.len()],
        &mut Vec::with_capacity(values.len()),
        &mut result,
        antisymmetric,
    );
    result
}
