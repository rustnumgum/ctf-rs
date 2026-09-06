// Adapted from cc4s CTF sparse summation key reindexing responsibilities.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Semiring, Wire}, diagonal::Projection, mapping::Distribution,
    sparse::SparseTensor, tensor::Tensor};

struct Indices {
    input: Projection,
    output: Projection,
    sources: Vec<Option<usize>>,
    broadcasts: usize,
}
impl Indices {
    fn new(input: &Distribution, labels_a: &str, output: &Distribution, labels_b: &str) -> Self {
        let input = Projection::new(&input.shape, labels_a);
        let output = Projection::new(&output.shape, labels_b);
        let mut broadcasts = 1;
        let sources = output.labels.bytes().enumerate().map(|(axis, label)| {
            let source = input.labels.bytes().position(|candidate| candidate == label);
            if let Some(source) = source { assert_eq!(input.shape[source], output.shape[axis]); }
            else { broadcasts *= output.shape[axis]; }
            source
        }).collect();
        Self { input, output, sources, broadcasts }
    }
    fn contributions<E: Clone>(&self, a: &Distribution, b: &Distribution,
        pairs: Vec<(usize, E)>) -> Vec<(usize, E)> {
        let mut result = Vec::new();
        for (key, value) in pairs {
            let Some(input) = self.input.project(&a.decode_key(key)) else { continue; };
            for mut broadcast in 0..self.broadcasts {
                let coordinates: Vec<_> = self.sources.iter().enumerate().map(|(axis, source)| {
                    if let Some(source) = source { input[*source] }
                    else {
                        let coordinate = broadcast % self.output.shape[axis];
                        broadcast /= self.output.shape[axis];
                        coordinate
                    }
                }).collect();
                result.push((b.encode_key(&self.output.expand(&coordinates)), value.clone()));
            }
        }
        result
    }
}

impl<A: Semiring> SparseTensor<'_, '_, A> where A::Element: Wire {
    /// Indexed sparse summation with explicit destination distribution.
    pub fn sum_from(&mut self, indices_b: &str, a: &Self, indices_a: &str,
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()));
        let indices = Indices::new(a.distribution(), indices_a, self.distribution(), indices_b);
        let pairs: Vec<_> = a.local_pairs().into_iter()
            .filter(|(key, _)| a.distribution().owner(*key) == a.context().rank())
            .map(|(key, value)| (key, self.algebra().multiply(&value, &alpha))).collect();
        let contributions = indices.contributions(a.distribution(), self.distribution(), pairs);
        // Only the indexed diagonal of B participates when labels repeat.
        self.scale_indexed(indices_b, &beta);
        self.write_add(&contributions);
    }

    /// Dense-to-sparse indexed summation. The source dense tensor is traversed
    /// locally; sparse output is never allocated as a dense staging tensor.
    pub fn sum_from_dense(&mut self, indices_b: &str, a: &Tensor<'_, '_, A>, indices_a: &str,
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()));
        let indices = Indices::new(a.distribution(), indices_a, self.distribution(), indices_b);
        let pairs: Vec<_> = a.local_pairs().into_iter()
            .filter(|(key, _)| a.distribution().owner(*key) == a.context().rank())
            // High-level summation sparsifies dense A before the sparse merge.
            .filter(|(_, value)| *value != a.algebra().zero())
            .map(|(key, value)| (key, self.algebra().multiply(&value, &alpha))).collect();
        let contributions = indices.contributions(a.distribution(), self.distribution(), pairs);
        self.scale_indexed(indices_b, &beta);
        self.write_add(&contributions);
    }
}

impl<A: Semiring + Clone> Tensor<'_, '_, A> where A::Element: Wire {
    pub fn sum_from_sparse(&mut self, indices_b: &str, a: &SparseTensor<'_, '_, A>, indices_a: &str,
        alpha: A::Element, beta: A::Element) {
        assert!(std::ptr::eq(self.context(), a.context()));
        let indices = Indices::new(a.distribution(), indices_a, self.distribution(), indices_b);
        let pairs: Vec<_> = a.local_pairs().into_iter()
            .filter(|(key, _)| a.distribution().owner(*key) == a.context().rank())
            .map(|(key, value)| {
                let value = if indices.input.labels.is_empty() {
                    self.algebra().multiply(&alpha, &value)
                } else { self.algebra().multiply(&value, &alpha) };
                (key, value)
            }).collect();
        let contributions = indices.contributions(a.distribution(), self.distribution(), pairs);
        let algebra = self.algebra().clone();
        self.transform_indexed(indices_b, |value| *value = algebra.multiply(&beta, value));
        self.write_add(&contributions);
    }
}
