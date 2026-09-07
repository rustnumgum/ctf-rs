// Adapted from cc4s CTF interface/{set,semiring}.h and sparse_formats/{csr,ccsr}.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Owned sparse matrix formats. IA, JA and COO coordinates retain upstream's
//! one-based indexing; no C++ packed-buffer ABI is retained. Explicit stored
//! zeros are preserved. COO conversion does not combine duplicate coordinates.

use crate::algebra::{Monoid, Semiring};

#[derive(Clone, Debug, PartialEq)]
pub struct Coo<T> {
    rows: usize,
    cols: usize,
    entries: Vec<(usize, usize, T)>,
}

impl<T: Clone> Coo<T> {
    pub fn new(rows: usize, cols: usize, entries: Vec<(usize, usize, T)>) -> Self {
        assert!(
            entries
                .iter()
                .all(|(r, c, _)| *r > 0 && *r <= rows && *c > 0 && *c <= cols)
        );
        Self {
            rows,
            cols,
            entries,
        }
    }
    pub fn entries(&self) -> &[(usize, usize, T)] {
        &self.entries
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn order(&self) -> Vec<usize> {
        let mut order: Vec<_> = (0..self.entries.len()).collect();
        // Source: column sort followed by stable row sort. Equal-coordinate
        // order is unspecified upstream; it does not coalesce duplicates.
        order.sort_by_key(|&i| self.entries[i].1);
        order.sort_by_key(|&i| self.entries[i].0);
        order
    }
    pub fn to_csr(&self) -> Csr<T> {
        let order = self.order();
        let mut ia = vec![0; self.rows + 1];
        ia[0] = 1;
        for &(r, _, _) in &self.entries {
            ia[r] += 1;
        }
        for r in 0..self.rows {
            ia[r + 1] += ia[r];
        }
        Csr {
            rows: self.rows,
            cols: self.cols,
            ia,
            ja: order.iter().map(|&i| self.entries[i].1).collect(),
            values: order.iter().map(|&i| self.entries[i].2.clone()).collect(),
        }
    }
    pub fn to_ccsr(&self) -> Ccsr<T> {
        let mut out = Ccsr {
            rows: self.rows,
            cols: self.cols,
            row_encoding: Vec::new(),
            ia: vec![1],
            ja: Vec::new(),
            values: Vec::new(),
        };
        for i in self.order() {
            let (r, c, v) = &self.entries[i];
            if out.row_encoding.last() != Some(r) {
                out.row_encoding.push(*r);
                out.ia.push(out.values.len() + 1);
            }
            out.ja.push(*c);
            out.values.push(v.clone());
            *out.ia.last_mut().unwrap() += 1;
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Csr<T> {
    rows: usize,
    cols: usize,
    ia: Vec<usize>,
    ja: Vec<usize>,
    values: Vec<T>,
}

impl<T: Clone> Csr<T> {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn row_offsets(&self) -> &[usize] {
        &self.ia
    }
    pub fn columns(&self) -> &[usize] {
        &self.ja
    }
    pub fn values(&self) -> &[T] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
    fn row(&self, r: usize) -> std::ops::Range<usize> {
        self.ia[r] - 1..self.ia[r + 1] - 1
    }
    fn empty(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            ia: vec![1],
            ja: Vec::new(),
            values: Vec::new(),
        }
    }
    fn append_row(&mut self, source: &Self, row: usize) {
        self.ja.extend_from_slice(&source.ja[source.row(row)]);
        self.values
            .extend_from_slice(&source.values[source.row(row)]);
        self.ia.push(self.values.len() + 1);
    }
    pub fn to_coo(&self) -> Coo<T> {
        let mut entries = Vec::with_capacity(self.values.len());
        for r in 0..self.rows {
            for i in self.row(r) {
                entries.push((r + 1, self.ja[i], self.values[i].clone()));
            }
        }
        Coo {
            rows: self.rows,
            cols: self.cols,
            entries,
        }
    }
    /// Source CSR_Matrix::partition: cyclic row strips, including empty parts.
    pub fn partition(&self, count: usize) -> Vec<Self> {
        assert!(count > 0);
        (0..count)
            .map(|p| {
                let mut part = Self::empty(
                    self.rows / count + usize::from(p < self.rows % count),
                    self.cols,
                );
                for r in (p..self.rows).step_by(count) {
                    part.append_row(self, r);
                }
                part
            })
            .collect()
    }
    pub fn assemble(parts: &[Self]) -> Self {
        assert!(!parts.is_empty());
        let rows: usize = parts.iter().map(|p| p.rows).sum();
        let mut out = Self::empty(rows, parts[0].cols);
        for (p, part) in parts.iter().enumerate() {
            assert_eq!(part.cols, out.cols);
            assert_eq!(
                part.rows,
                rows / parts.len() + usize::from(p < rows % parts.len())
            );
        }
        for r in 0..rows {
            out.append_row(&parts[r % parts.len()], r / parts.len());
        }
        out
    }
    /// Source csr_add's symbolic union and reverse-column numeric scatter.
    /// Inputs must have unique coordinates (COO conversion alone does not ensure this).
    pub fn add<A: Monoid<Element = T>>(&self, b: &Self, algebra: &A) -> Self {
        assert_eq!(self.rows, b.rows);
        for matrix in [self, b] {
            for r in 0..matrix.rows {
                assert!(matrix.ja[matrix.row(r)].windows(2).all(|x| x[0] < x[1]));
            }
        }
        let cols = self.cols.max(b.cols);
        let mut out = Self::empty(self.rows, cols);
        let mut present = vec![false; cols];
        let mut reverse = vec![0; cols];
        for r in 0..self.rows {
            present.fill(false);
            for matrix in [self, b] {
                for i in matrix.row(r) {
                    present[matrix.ja[i] - 1] = true;
                }
            }
            for c in 0..cols {
                if present[c] {
                    reverse[c] = out.values.len();
                    out.ja.push(c + 1);
                    out.values.push(algebra.zero());
                }
            }
            present.fill(false);
            for i in self.row(r) {
                let c = self.ja[i] - 1;
                out.values[reverse[c]] = self.values[i].clone();
                present[c] = true;
            }
            for i in b.row(r) {
                let c = b.ja[i] - 1;
                let at = reverse[c];
                out.values[at] = if present[c] {
                    algebra.add(&out.values[at], &b.values[i])
                } else {
                    b.values[i].clone()
                };
            }
            out.ia.push(out.values.len() + 1);
        }
        out
    }
    /// gen_csrmm: dense B and C are column-major, with no leading padding.
    pub fn multiply_dense<A: Semiring<Element = T>>(
        &self,
        n: usize,
        b: &[T],
        alpha: &T,
        beta: &T,
        c: &mut [T],
        algebra: &A,
    ) {
        assert_eq!(b.len(), self.cols * n);
        assert_eq!(c.len(), self.rows * n);
        for r in 0..self.rows {
            for col in 0..n {
                let at = col * self.rows + r;
                c[at] = algebra.multiply(beta, &c[at]);
                let mut indices = self.row(r);
                if let Some(first) = indices.next() {
                    let mut sum = algebra.multiply(
                        &self.values[first],
                        &b[col * self.cols + self.ja[first] - 1],
                    );
                    for i in indices {
                        sum = algebra.add(
                            &sum,
                            &algebra
                                .multiply(&self.values[i], &b[col * self.cols + self.ja[i] - 1]),
                        );
                    }
                    c[at] = algebra.add(&c[at], &algebra.multiply(alpha, &sum));
                }
            }
        }
    }
    /// gen_csrmultcsr: symbolic row counts, dense row accumulators, sorted
    /// structural output, alpha scaling, then beta*C sparse addition. Cancellation
    /// does not drop structural entries. No dense m*n intermediate is allocated.
    pub fn multiply_sparse<A: Semiring<Element = T>>(
        &self,
        b: &Self,
        alpha: &T,
        beta: &T,
        c: Option<&Self>,
        algebra: &A,
    ) -> Self
    where
        T: PartialEq,
    {
        assert_eq!(self.cols, b.rows);
        if let Some(c) = c {
            assert_eq!(c.shape(), (self.rows, b.cols));
        }
        let mut out = Self::empty(self.rows, b.cols);
        let mut present = vec![false; b.cols];
        for r in 0..self.rows {
            present.fill(false);
            for i in self.row(r) {
                for j in b.row(self.ja[i] - 1) {
                    present[b.ja[j] - 1] = true;
                }
            }
            out.ia
                .push(out.ia[r] + present.iter().filter(|&&x| x).count());
        }
        out.values = vec![algebra.zero(); out.ia[self.rows] - 1];
        out.ja = vec![0; out.values.len()];
        let mut accum = vec![algebra.zero(); b.cols];
        for r in 0..self.rows {
            present.fill(false);
            accum.fill(algebra.zero());
            for i in self.row(r) {
                for j in b.row(self.ja[i] - 1) {
                    let col = b.ja[j] - 1;
                    present[col] = true;
                    accum[col] = algebra.add(
                        &accum[col],
                        &algebra.multiply(&self.values[i], &b.values[j]),
                    );
                }
            }
            let mut at = out.ia[r] - 1;
            for col in 0..b.cols {
                if present[col] {
                    out.ja[at] = col + 1;
                    out.values[at] = algebra.multiply(alpha, &accum[col]);
                    at += 1;
                }
            }
        }
        if *beta != algebra.zero() {
            if let Some(c) = c {
                if !c.values.is_empty() {
                    let mut scaled = c.clone();
                    for v in &mut scaled.values {
                        *v = algebra.multiply(beta, v);
                    }
                    return scaled.add(&out, algebra);
                }
            }
        }
        out
    }
}

/// Doubly compressed CSR: only represented rows have IA entries. Row encodings
/// remain one-based global row numbers, so storage is independent of empty rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Ccsr<T> {
    rows: usize,
    cols: usize,
    row_encoding: Vec<usize>,
    ia: Vec<usize>,
    ja: Vec<usize>,
    values: Vec<T>,
}
impl<T: Clone> Ccsr<T> {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn row_encoding(&self) -> &[usize] {
        &self.row_encoding
    }
    pub fn row_offsets(&self) -> &[usize] {
        &self.ia
    }
    pub fn columns(&self) -> &[usize] {
        &self.ja
    }
    pub fn values(&self) -> &[T] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
    fn row(&self, r: usize) -> std::ops::Range<usize> {
        self.ia[r] - 1..self.ia[r + 1] - 1
    }
    fn empty(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_encoding: Vec::new(),
            ia: vec![1],
            ja: Vec::new(),
            values: Vec::new(),
        }
    }
    fn append_row(&mut self, source: &Self, row: usize, encoding: usize) {
        self.row_encoding.push(encoding);
        self.ja.extend_from_slice(&source.ja[source.row(row)]);
        self.values
            .extend_from_slice(&source.values[source.row(row)]);
        self.ia.push(self.values.len() + 1);
    }
    pub fn to_coo(&self) -> Coo<T> {
        let mut entries = Vec::with_capacity(self.values.len());
        for (r, &encoding) in self.row_encoding.iter().enumerate() {
            for i in self.row(r) {
                entries.push((encoding, self.ja[i], self.values[i].clone()));
            }
        }
        Coo {
            rows: self.rows,
            cols: self.cols,
            entries,
        }
    }
    pub fn partition(&self, count: usize) -> Vec<Self> {
        assert!(count > 0);
        let mut parts: Vec<_> = (0..count)
            .map(|p| {
                Self::empty(
                    self.rows / count + usize::from(p < self.rows % count),
                    self.cols,
                )
            })
            .collect();
        for (r, &encoding) in self.row_encoding.iter().enumerate() {
            parts[(encoding - 1) % count].append_row(self, r, (encoding - 1) / count + 1);
        }
        parts
    }
    /// Source CCSR_Matrix::assemble: merge compressed row streams; equal local
    /// row numbers are selected in part order before restoring global encoding.
    pub fn assemble(parts: &[Self]) -> Self {
        assert!(!parts.is_empty());
        let rows: usize = parts.iter().map(|p| p.rows).sum();
        let mut out = Self::empty(rows, parts[0].cols);
        for (p, part) in parts.iter().enumerate() {
            assert_eq!(part.cols, out.cols);
            assert_eq!(
                part.rows,
                rows / parts.len() + usize::from(p < rows % parts.len())
            );
        }
        let mut cursor = vec![0; parts.len()];
        for _ in 0..parts.iter().map(|p| p.row_encoding.len()).sum::<usize>() {
            let p = (0..parts.len())
                .filter(|&p| cursor[p] < parts[p].row_encoding.len())
                .min_by_key(|&p| (parts[p].row_encoding[cursor[p]], p))
                .unwrap();
            let r = cursor[p];
            out.append_row(
                &parts[p],
                r,
                (parts[p].row_encoding[r] - 1) * parts.len() + p + 1,
            );
            cursor[p] += 1;
        }
        out
    }

    /// Merge represented row streams; intersecting rows use the same symbolic
    /// union/reverse scatter as CSR. Never expand the full logical row dimension.
    pub fn add<A: Monoid<Element = T>>(&self, b: &Self, algebra: &A) -> Self {
        assert_eq!(self.rows, b.rows);
        let mut out = Self::empty(self.rows, self.cols.max(b.cols));
        let (mut i, mut j) = (0, 0);
        while i < self.row_encoding.len() || j < b.row_encoding.len() {
            if j == b.row_encoding.len()
                || (i < self.row_encoding.len() && self.row_encoding[i] < b.row_encoding[j])
            {
                out.append_row(self, i, self.row_encoding[i]);
                i += 1;
            } else if i == self.row_encoding.len() || b.row_encoding[j] < self.row_encoding[i] {
                out.append_row(b, j, b.row_encoding[j]);
                j += 1;
            } else {
                let row_matrix = |matrix: &Self, row| Csr {
                    rows: 1,
                    cols: matrix.cols,
                    ia: vec![1, matrix.row(row).len() + 1],
                    ja: matrix.ja[matrix.row(row)].to_vec(),
                    values: matrix.values[matrix.row(row)].to_vec(),
                };
                let row = row_matrix(self, i).add(&row_matrix(b, j), algebra);
                out.row_encoding.push(self.row_encoding[i]);
                out.ja.extend(row.ja);
                out.values.extend(row.values);
                out.ia.push(out.values.len() + 1);
                i += 1;
                j += 1;
            }
        }
        out
    }

    /// gen_ccsrmm: CSR multiply on the represented rows followed by a local
    /// column-to-row transpose and sparse beta*C merge. The source omits one
    /// all-zero final B column as padding, but retains zeros in all other columns.
    pub fn multiply_dense<A: Semiring<Element = T>>(
        &self,
        n: usize,
        b: &[T],
        alpha: &T,
        beta: &T,
        c: Option<&Self>,
        algebra: &A,
    ) -> Self
    where
        T: PartialEq,
    {
        assert_eq!(b.len(), self.cols * n);
        if let Some(c) = c {
            assert_eq!(c.shape(), (self.rows, n));
        }
        let columns = if n > 0
            && b[(n - 1) * self.cols..]
                .iter()
                .all(|v| *v == algebra.zero())
        {
            n - 1
        } else {
            n
        };
        let mut out = Self::empty(self.rows, n);
        if columns > 0 {
            let packed = Csr {
                rows: self.row_encoding.len(),
                cols: self.cols,
                ia: self.ia.clone(),
                ja: self.ja.clone(),
                values: self.values.clone(),
            };
            let mut dense = vec![algebra.zero(); packed.rows * columns];
            packed.multiply_dense(
                columns,
                &b[..self.cols * columns],
                alpha,
                &algebra.one(),
                &mut dense,
                algebra,
            );
            for (r, &encoding) in self.row_encoding.iter().enumerate() {
                out.row_encoding.push(encoding);
                for col in 0..columns {
                    out.ja.push(col + 1);
                    out.values.push(dense[col * packed.rows + r].clone());
                }
                out.ia.push(out.values.len() + 1);
            }
        }
        if *beta != algebra.zero() {
            if let Some(c) = c {
                let mut scaled = c.clone();
                for v in &mut scaled.values {
                    *v = algebra.multiply(beta, v);
                }
                return scaled.add(&out, algebra);
            }
        }
        out
    }
}
