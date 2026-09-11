// Source-compatible sparse random generation from interface/tensor.cxx.

use crate::{
    algebra::{Arithmetic, Semiring},
    random::Generator,
};

use super::SparseSymmetricTensor;

fn local_generation_count(
    total_size: usize,
    fraction: f64,
    processes: usize,
    rank: usize,
) -> usize {
    let mut term = total_size as f64 * fraction;
    let mut expected = 0.0;
    for divisor in 2..20 {
        expected += term;
        term *= fraction / divisor as f64;
    }
    let generated = (expected + 0.5) as usize;
    generated / processes + usize::from(generated % processes > rank)
}

impl SparseSymmetricTensor<'_, '_, Arithmetic<f32>> {
    /// Fill sparse canonical storage from raw rectangular random keys. Raw
    /// symmetry-equivalent writes first collide additively, then every stored
    /// value is replaced by the source's independently sampled value.
    pub fn fill_random_sparse(
        &mut self,
        minimum: f32,
        maximum: f32,
        fraction: f64,
        generator: &mut Generator,
    ) {
        let total_size = self
            .distribution
            .distribution()
            .shape
            .iter()
            .product::<usize>();
        let generated = local_generation_count(
            total_size,
            fraction,
            self.context().size(),
            self.context().rank(),
        );
        self.storage.sparsify(|_| false);
        let one = self.algebra().one();
        let mut candidates = Vec::with_capacity(generated);
        for _ in 0..generated {
            let key = (generator.unit_interval() * total_size as f64) as usize;
            candidates.push((key, one));
        }
        self.write_add(&candidates);

        let span = maximum - minimum;
        self.storage.transform_stored(|_, value| {
            let draw = generator.unit_interval() as f32;
            let indicator = if *value != 0.0 { 1.0 } else { 0.0 };
            *value = indicator * (draw * span + minimum);
        });
    }
}
