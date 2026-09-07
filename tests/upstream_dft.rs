//! Pinned test/dft.cxx: symmetric complex DFT product and custom scalar reduction.
//! Rust uses native usize coordinates, not the excluded int/int64 C++ ABI overloads.
use ctf::{
    algebra::{Arithmetic, Complex},
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{NS, SY},
};

type Scalar = Complex<f64>;

fn make<'c, 'r>(context: &'c Context<'r>, n: Option<usize>)
    -> SymmetricTensor<'c, 'r, Arithmetic<Scalar>> {
    let topology = Topology::new(vec![context.size()]);
    let (shape, links, mappings) = if let Some(n) = n {
        let mut mappings = vec![Mapping::Unmapped; 2];
        mappings[0].augment_physical(&topology, 0);
        for mapping in &mut mappings { mapping.augment_virtual(2 * context.size()); }
        (vec![n, n], vec![SY, NS], mappings)
    } else { (vec![], vec![], vec![]) };
    SymmetricTensor::new(context,
        SymmetricDistribution::new(Distribution::new(shape, topology, mappings), links),
        Arithmetic::new())
}

fn run(context: &Context<'_>) {
    let n = 8;
    let mut dft = make(context, Some(n));
    let mut inverse = make(context, Some(n));
    dft.transform(|key, value| {
        let angle = -2.0 * (key % n) as f64 * (key / n) as f64
            * (std::f64::consts::PI / n as f64);
        *value = Scalar::new(angle.cos(), angle.sin());
    });
    inverse.transform(|key, value| {
        let angle = 2.0 * (key % n) as f64 * (key / n) as f64
            * (std::f64::consts::PI / n as f64);
        *value = Scalar::new(angle.cos() / n as f64, angle.sin() / n as f64);
    });
    let operand = dft.redistribute(dft.distribution().clone());
    let topology = Topology::new(vec![context.size()]);
    dft.contract_from_on("ik", &operand, "ij", &inverse, "jk",
        topology.clone(), "j", Scalar::new(0.5, 0.0), Scalar::new(0.0, 0.0), true).unwrap();

    // This source operation is active although ss has no numeric assertion.
    let mut ss = make(context, None);
    ss.contract_function_from_on("", &dft, "ij", &dft, "ij", topology, "i",
        Scalar::new(1.0, 0.0), Scalar::new(0.0, 0.0), true,
        |a, b| Scalar::new(a.re + b.re, a.im + b.im)).unwrap();
    for (key, value) in dft.local_pairs() {
        assert!(value.re.is_finite() && value.im.is_finite());
        let expected = if key % n == key / n { 1.0 } else { 0.0 };
        assert!((value.re - expected).abs() < 1e-9);
    }
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let child = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&child);
    child.close();
    if world.rank() == 0 {
        println!("DIGIT / PASS upstream_dft: SY complex DFT/inverse and custom scalar sum; original real error <1e-9; world+parity");
    }
    world.close();
    runtime.finalize();
}
