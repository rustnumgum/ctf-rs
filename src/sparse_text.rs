//! Sparse coordinate text I/O adapted from `interface/graph_io_aux.cxx:1-150`
//! and `interface/tensor.cxx:945-1015`; MPI chunking is delegated to Context.

use std::{fmt::Write, path::Path};

use crate::{
    algebra::{Arithmetic, Monoid, Semiring},
    mapping::Distribution,
    sparse::SparseTensor,
    symmetric_tensor::SymmetricTensor,
    tensor::Tensor,
};

trait SparseTextScalar: Clone + PartialEq {
    fn parse_text(value: &str) -> Self;
    fn format_text(&self) -> String;
}

impl SparseTextScalar for f32 {
    fn parse_text(value: &str) -> Self { value.parse().unwrap() }
    fn format_text(&self) -> String { format!("{self:.6}") }
}

impl SparseTextScalar for f64 {
    fn parse_text(value: &str) -> Self { value.parse().unwrap() }
    fn format_text(&self) -> String { format!("{self:.6}") }
}

impl SparseTextScalar for i32 {
    fn parse_text(value: &str) -> Self { value.parse().unwrap() }
    fn format_text(&self) -> String { self.to_string() }
}

impl SparseTextScalar for i64 {
    fn parse_text(value: &str) -> Self { value.parse().unwrap() }
    fn format_text(&self) -> String { self.to_string() }
}

fn parse_sparse_text<T: SparseTextScalar>(
    bytes: &[u8],
    distribution: &Distribution,
    algebra: &Arithmetic<T>,
    with_values: bool,
    reverse_order: bool,
) -> Vec<(usize, T)>
where
    Arithmetic<T>: Semiring<Element = T>,
{
    let text = std::str::from_utf8(bytes).unwrap();
    let order = distribution.shape.len();
    text.lines()
        .map(|line| {
            let mut fields = line.split_whitespace();
            let mut coordinates: Vec<usize> = (0..order)
                .map(|_| fields.next().unwrap().parse().unwrap())
                .collect();
            let value = if with_values {
                T::parse_text(fields.next().unwrap())
            } else {
                algebra.one()
            };
            assert!(fields.next().is_none());
            if reverse_order {
                coordinates.reverse();
            }
            (distribution.encode_key(&coordinates), value)
        })
        .collect()
}

fn serialize_sparse_text<T: SparseTextScalar>(
    distribution: &Distribution,
    pairs: &[(usize, T)],
    with_values: bool,
    reverse_order: bool,
) -> Vec<u8> {
    let mut output = String::new();
    for (key, value) in pairs {
        let mut coordinates = distribution.decode_key(*key);
        if reverse_order {
            coordinates.reverse();
        }
        for (index, coordinate) in coordinates.iter().enumerate() {
            if index != 0 {
                output.push(' ');
            }
            write!(output, "{coordinate}").unwrap();
        }
        if with_values {
            output.push(' ');
            output.push_str(&value.format_text());
        }
        output.push('\n');
    }
    output.into_bytes()
}

macro_rules! sparse_text_apis {
    ($($scalar:ty),*) => {
        $(
        impl<'c, 'r> Tensor<'c, 'r, Arithmetic<$scalar>> {
            pub fn read_sparse_from_file(
                &mut self,
                path: &Path,
                with_values: bool,
                reverse_order: bool,
            ) {
                let bytes = self.context().inner.read_sparse_text(path);
                let algebra = Arithmetic::<$scalar>::new();
                let pairs = parse_sparse_text(
                    &bytes,
                    self.distribution(),
                    &algebra,
                    with_values,
                    reverse_order,
                );
                self.write_add(&pairs);
            }

            pub fn write_sparse_to_file(
                &self,
                path: &Path,
                with_values: bool,
                reverse_order: bool,
            ) {
                let algebra = Arithmetic::<$scalar>::new();
                let rank = self.context().rank();
                let pairs: Vec<_> = self
                    .local_pairs()
                    .into_iter()
                    .filter(|(key, value)| {
                        self.distribution().owner(*key) == rank && value != &algebra.zero()
                    })
                    .collect();
                let bytes = serialize_sparse_text(
                    self.distribution(),
                    &pairs,
                    with_values,
                    reverse_order,
                );
                self.context().inner.write_sparse_text(path, &bytes);
            }
        }

        impl<'c, 'r> SparseTensor<'c, 'r, Arithmetic<$scalar>> {
            pub fn read_sparse_from_file(
                &mut self,
                path: &Path,
                with_values: bool,
                reverse_order: bool,
            ) {
                let bytes = self.context().inner.read_sparse_text(path);
                let algebra = Arithmetic::<$scalar>::new();
                let pairs = parse_sparse_text(
                    &bytes,
                    self.distribution(),
                    &algebra,
                    with_values,
                    reverse_order,
                );
                self.write_add(&pairs);
            }

            pub fn write_sparse_to_file(
                &self,
                path: &Path,
                with_values: bool,
                reverse_order: bool,
            ) {
                let rank = self.context().rank();
                let pairs: Vec<_> = self
                    .local_pairs()
                    .into_iter()
                    .filter(|(key, _)| self.distribution().owner(*key) == rank)
                    .collect();
                let bytes = serialize_sparse_text(
                    self.distribution(),
                    &pairs,
                    with_values,
                    reverse_order,
                );
                self.context().inner.write_sparse_text(path, &bytes);
            }
        }

        impl<'c, 'r> SymmetricTensor<'c, 'r, Arithmetic<$scalar>> {
            pub fn read_sparse_from_file(
                &mut self,
                path: &Path,
                with_values: bool,
                reverse_order: bool,
            ) {
                let bytes = self.context().inner.read_sparse_text(path);
                let algebra = Arithmetic::<$scalar>::new();
                let pairs = parse_sparse_text(
                    &bytes,
                    self.distribution().distribution(),
                    &algebra,
                    with_values,
                    reverse_order,
                );
                self.write_add(&pairs);
            }

            pub fn write_sparse_to_file(
                &self,
                path: &Path,
                with_values: bool,
                reverse_order: bool,
            ) {
                let algebra = Arithmetic::<$scalar>::new();
                let rank = self.context().rank();
                let pairs: Vec<_> = self
                    .local_pairs()
                    .into_iter()
                    .filter(|(key, value)| {
                        self.distribution().distribution().owner(*key) == rank
                            && value != &algebra.zero()
                    })
                    .collect();
                let bytes = serialize_sparse_text(
                    self.distribution().distribution(),
                    &pairs,
                    with_values,
                    reverse_order,
                );
                self.context().inner.write_sparse_text(path, &bytes);
            }
        }
        )*
    };
}

sparse_text_apis!(f32, f64, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::{Mapping, Topology};

    fn distribution() -> Distribution {
        Distribution::new(
            vec![3, 4, 2],
            Topology::new(vec![1]),
            vec![Mapping::Unmapped; 3],
        )
    }

    #[test]
    fn coordinates_and_values_roundtrip() {
        let distribution = distribution();
        let key = distribution.encode_key(&[1, 2, 1]);
        let pairs = vec![(key, 7.25_f64)];
        let bytes = serialize_sparse_text(&distribution, &pairs, true, false);
        assert_eq!(bytes, b"1 2 1 7.250000\n");
        assert_eq!(
            parse_sparse_text(
                &bytes,
                &distribution,
                &Arithmetic::<f64>::new(),
                true,
                false,
            ),
            pairs,
        );
    }

    #[test]
    fn no_values_use_multiplicative_identity() {
        let distribution = distribution();
        let key = distribution.encode_key(&[2, 1, 0]);
        let bytes = serialize_sparse_text(&distribution, &[(key, 1_i32)], false, false);
        assert_eq!(bytes, b"2 1 0\n");
        assert_eq!(
            parse_sparse_text(
                &bytes,
                &distribution,
                &Arithmetic::<i32>::new(),
                false,
                false,
            ),
            vec![(key, 1)],
        );
    }

    #[test]
    fn reverse_order_roundtrip() {
        let distribution = distribution();
        let key = distribution.encode_key(&[1, 2, 0]);
        let bytes = serialize_sparse_text(&distribution, &[(key, 9_i64)], true, true);
        assert_eq!(bytes, b"0 2 1 9\n");
        assert_eq!(
            parse_sparse_text(
                &bytes,
                &distribution,
                &Arithmetic::<i64>::new(),
                true,
                true,
            ),
            vec![(key, 9)],
        );
    }

    #[test]
    fn typed_float_format_has_six_fractional_digits() {
        assert_eq!(1.25_f32.format_text(), "1.250000");
        assert_eq!((-0.5_f64).format_text(), "-0.500000");
    }

    #[test]
    fn integer_roundtrip() {
        let distribution = distribution();
        let pairs = vec![
            (distribution.encode_key(&[0, 0, 0]), -17_i32),
            (distribution.encode_key(&[2, 3, 1]), 123456789_i32),
        ];
        let bytes = serialize_sparse_text(&distribution, &pairs, true, false);
        assert_eq!(
            parse_sparse_text(
                &bytes,
                &distribution,
                &Arithmetic::<i32>::new(),
                true,
                false,
            ),
            pairs,
        );
    }
}
