//! Dense nonsymmetric local transpose.
//!
//! `order` uses the CTF convention: it lists original axes from fastest to
//! slowest in the destination buffer.  The implementation copies the largest
//! leading run which is contiguous in both layouts instead of transposing one
//! element at a time.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransposePlan {
    shape: Vec<usize>,
    order: Vec<usize>,
    source_strides: Vec<usize>,
    destination_strides: Vec<usize>,
    contiguous_len: usize,
    len: usize,
}

impl TransposePlan {
    pub fn new(shape: &[usize], order: &[usize]) -> Self {
        assert_eq!(shape.len(), order.len());
        let mut seen = vec![false; order.len()];
        for &axis in order {
            assert!(axis < order.len() && !seen[axis]);
            seen[axis] = true;
        }

        let source_strides = column_major_strides(shape);
        let mut destination_strides = vec![0; shape.len()];
        let mut stride = 1;
        for &axis in order {
            destination_strides[axis] = stride;
            stride *= shape[axis];
        }
        let len = shape.iter().product();
        let mut contiguous_len = 1;
        for (position, &axis) in order.iter().enumerate() {
            if position != axis {
                break;
            }
            contiguous_len *= shape[axis];
        }

        Self {
            shape: shape.to_vec(),
            order: order.to_vec(),
            source_strides,
            destination_strides,
            contiguous_len,
            len,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn order(&self) -> &[usize] {
        &self.order
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of elements copied by each inner contiguous operation.
    pub fn contiguous_len(&self) -> usize {
        self.contiguous_len
    }

    pub fn execute<T: Clone>(&self, input: &[T], direction: Direction) -> Vec<T> {
        assert_eq!(input.len(), self.len);
        let mut output = input.to_vec();
        self.execute_into(input, &mut output, direction);
        output
    }

    pub fn execute_into<T: Clone>(
        &self,
        input: &[T],
        output: &mut [T],
        direction: Direction,
    ) {
        assert_eq!(input.len(), self.len);
        assert_eq!(output.len(), self.len);
        if self.len == 0 {
            return;
        }

        for source_base in (0..self.len).step_by(self.contiguous_len) {
            let destination_base = self.destination_offset(source_base);
            let (read, write) = match direction {
                Direction::Forward => (source_base, destination_base),
                Direction::Backward => (destination_base, source_base),
            };
            output[write..write + self.contiguous_len]
                .clone_from_slice(&input[read..read + self.contiguous_len]);
        }
    }

    fn destination_offset(&self, source_offset: usize) -> usize {
        self.shape
            .iter()
            .enumerate()
            .map(|(axis, &extent)| {
                ((source_offset / self.source_strides[axis]) % extent)
                    * self.destination_strides[axis]
            })
            .sum()
    }
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape
        .iter()
        .map(|&extent| {
            let current = stride;
            stride *= extent;
            current
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Direction, TransposePlan};

    #[test]
    fn copies_maximal_contiguous_prefix_and_round_trips() {
        let plan = TransposePlan::new(&[3, 2, 4], &[0, 2, 1]);
        assert_eq!(plan.contiguous_len(), 3);
        let input: Vec<_> = (0..24).collect();
        let transposed = plan.execute(&input, Direction::Forward);
        let expected: Vec<_> = (0..24)
            .map(|offset| {
                let i = offset % 3;
                let k = (offset / 3) % 4;
                let j = offset / 12;
                input[i + 3 * j + 6 * k]
            })
            .collect();
        assert_eq!(transposed, expected);
        assert_eq!(plan.execute(&transposed, Direction::Backward), input);
    }

    #[test]
    fn noncontiguous_scalar_and_empty_shapes_are_exact() {
        let plan = TransposePlan::new(&[2, 3], &[1, 0]);
        assert_eq!(plan.contiguous_len(), 1);
        assert_eq!(
            plan.execute(&[0, 1, 2, 3, 4, 5], Direction::Forward),
            [0, 2, 4, 1, 3, 5]
        );
        assert_eq!(TransposePlan::new(&[], &[]).execute(&[7], Direction::Forward), [7]);
        assert!(TransposePlan::new(&[0, 3], &[1, 0])
            .execute::<i32>(&[], Direction::Forward)
            .is_empty());
    }
}
