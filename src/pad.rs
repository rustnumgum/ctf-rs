//! Canonical packed padding metadata and storage conversion.
//!
//! The pinned source's `pad_tsr` implementation is disabled and its virtual
//! nonsymmetric zeroing loop advances the block counter twice.  Neither defect
//! is reproduced here: one explicit canonical-offset plan drives padding,
//! depadding, and zeroing for every virtual block.

use crate::algebra::Monoid;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalPadding {
    allocated_len: usize,
    canonical_offsets: Vec<usize>,
}

impl CanonicalPadding {
    pub fn new(allocated_len: usize, canonical_offsets: Vec<usize>) -> Self {
        assert!(canonical_offsets.iter().all(|&offset| offset < allocated_len));
        assert!(canonical_offsets.windows(2).all(|pair| pair[0] < pair[1]));
        Self {
            allocated_len,
            canonical_offsets,
        }
    }

    pub fn allocated_len(&self) -> usize {
        self.allocated_len
    }

    pub fn canonical_len(&self) -> usize {
        self.canonical_offsets.len()
    }

    pub fn canonical_offsets(&self) -> &[usize] {
        &self.canonical_offsets
    }

    pub fn padding_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let mut canonical = self.canonical_offsets.iter().copied().peekable();
        (0..self.allocated_len).filter(move |offset| {
            if canonical.peek().copied() == Some(*offset) {
                canonical.next();
                false
            } else {
                true
            }
        })
    }

    pub fn pad<A: Monoid>(&self, algebra: &A, compact: &[A::Element]) -> Vec<A::Element> {
        assert_eq!(compact.len(), self.canonical_len());
        let mut padded = vec![algebra.zero(); self.allocated_len];
        for (&offset, value) in self.canonical_offsets.iter().zip(compact) {
            padded[offset] = value.clone();
        }
        padded
    }

    pub fn depad<T: Clone>(&self, padded: &[T]) -> Vec<T> {
        assert_eq!(padded.len(), self.allocated_len);
        self.canonical_offsets
            .iter()
            .map(|&offset| padded[offset].clone())
            .collect()
    }

    pub fn zero_padding<A: Monoid>(&self, algebra: &A, padded: &mut [A::Element]) {
        assert_eq!(padded.len(), self.allocated_len);
        for offset in self.padding_offsets() {
            padded[offset] = algebra.zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CanonicalPadding;
    use crate::algebra::Arithmetic;

    #[test]
    fn canonical_pad_depad_and_zero_are_exact() {
        let plan = CanonicalPadding::new(10, vec![0, 2, 3, 7, 9]);
        assert_eq!(plan.padding_offsets().collect::<Vec<_>>(), [1, 4, 5, 6, 8]);
        let padded = plan.pad(&Arithmetic::<i64>::new(), &[4, 5, 6, 7, 8]);
        assert_eq!(padded, [4, 0, 5, 6, 0, 0, 0, 7, 0, 8]);
        assert_eq!(plan.depad(&padded), [4, 5, 6, 7, 8]);
        let mut dirty = vec![99i64; 10];
        plan.zero_padding(&Arithmetic::<i64>::new(), &mut dirty);
        assert_eq!(dirty, [99, 0, 99, 99, 0, 0, 0, 99, 0, 99]);
    }

    #[test]
    fn pinned_virtual_block_skip_defect_is_not_preserved() {
        let plan = CanonicalPadding::new(12, vec![0, 2, 4, 6, 8, 10]);
        let mut blocks = vec![7i32; 12];
        plan.zero_padding(&Arithmetic::<i32>::new(), &mut blocks);
        assert_eq!(blocks, [7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0]);
    }

    #[test]
    fn scalar_and_empty_allocations_are_distinct() {
        let scalar = CanonicalPadding::new(1, vec![0]);
        assert_eq!(scalar.pad(&Arithmetic::<i64>::new(), &[3]), [3]);
        let empty = CanonicalPadding::new(0, vec![]);
        assert!(empty.pad(&Arithmetic::<i64>::new(), &[]).is_empty());
    }
}
