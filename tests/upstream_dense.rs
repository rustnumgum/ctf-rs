// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted test/diag_ctr.cxx and test/reduce_bcast.cxx at the pinned commit.
use ctf::{
    algebra::Arithmetic,
    context::Runtime,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};
struct Drand48(u64);
impl Drand48 {
    fn seeded(seed: u64) -> Self {
        Self((seed << 16) | 0x330e)
    }
    fn next(&mut self) -> f64 {
        self.0 = (self.0.wrapping_mul(0x5deece66d).wrapping_add(11)) & ((1 << 48) - 1);
        self.0 as f64 / (1u64 << 48) as f64
    }
}
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let np = world.size();
    let topo = Topology::new(if np == 4 { vec![2, 2] } else { vec![np] });
    let make = |shape: Vec<usize>| {
        Tensor::new(
            &world,
            Distribution::cyclic(shape, np),
            Arithmetic::<f64>::new(),
        )
    };
    let (n, m) = (6, 7);
    let mut rng = Drand48::seeded(13 * world.rank() as u64);
    let mut a = make(vec![n, m, n, m]);
    a.transform(|_, v| *v = rng.next() - 0.5);
    let mut scalar = make(vec![]);
    scalar
        .sum_from("", &a, "aiai", topo.clone(), 1., 0.)
        .unwrap();
    let trace = scalar.reduce();
    assert!(trace.is_finite() && trace.abs() >= 1e-10);
    let mut ma = make(vec![n, m]);
    ma.sum_from("ai", &a, "aiai", topo.clone(), 1., 0.).unwrap();
    let difference = (trace - ma.reduce()).abs();
    assert!(difference.is_finite() && difference <= 1e-10);
    if world.rank() == 0 {
        println!("DIGIT / PASS upstream diag_ctr: delta={difference:e}, bound=1e-10");
    }
    let mut b = make(vec![n, 1]);
    b.transform(|_, v| *v = rng.next());
    let mut c = make(vec![n, n]);
    c.transform(|_, v| *v = rng.next());
    let mut c2 = make(vec![n, n]);
    c2.sum_from("ij", &c, "ij", topo.clone(), 1., 0.).unwrap();
    let mut d = make(vec![n]);
    for (iteration, labels) in ["ij", "ik"].iter().enumerate() {
        if iteration != 0 {
            c.sum_from("ij", &c2, "ij", topo.clone(), 1., 0.).unwrap();
        }
        c.sum_from("ij", &b, "ik", topo.clone(), 1., 1.).unwrap();
        d.sum_from("i", &b, labels, topo.clone(), 1., 0.).unwrap();
        c2.sum_from("ij", &d, "i", topo.clone(), 1., 1.).unwrap();
        c.sum_from("ij", &c2, "ij", topo.clone(), -1., 1.).unwrap();
        c.transform(|_, v| *v *= *v);
        let norm = c.reduce().sqrt();
        assert!(norm.is_finite() && norm <= 1e-6);
        if world.rank() == 0 {
            println!(
                "DIGIT / PASS upstream reduce_bcast variant {iteration}: Frobenius={norm:e}, bound=1e-6"
            );
        }
    }
    drop(d);
    drop(c2);
    drop(c);
    drop(b);
    drop(ma);
    drop(scalar);
    drop(a);
    if world.rank() == 0 {
        println!("upstream_dense complete: ranks={np}");
    }
    world.close();
    runtime.finalize();
}
