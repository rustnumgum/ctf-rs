// Pinned examples/dft_3D.cxx: three-dimensional DFT by contractions.

use ctf::{
    algebra::{Arithmetic, Complex, CustomMonoid},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{NS, SY},
};

type Scalar = Complex<f64>;
type Tensor<'c, 'r> = SymmetricTensor<'c, 'r, Arithmetic<Scalar>>;

fn cis(angle: f64) -> Scalar {
    Scalar::new(angle.cos(), angle.sin())
}

fn distribution(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !shape.is_empty() {
        mappings[0].augment_physical(&topology, 0);
    }
    Distribution::new(shape, topology, mappings)
}

fn tensor<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    links: Vec<ctf::symmetry::Symmetry>,
) -> Tensor<'c, 'r> {
    let mut distribution = distribution(context, shape);
    if links.iter().any(|&link| link != NS) {
        for mapping in &mut distribution.mappings {
            mapping.augment_virtual(2 * context.size());
        }
    }
    Tensor::new(
        context,
        SymmetricDistribution::new(distribution, links),
        Arithmetic::new(),
    )
}

fn run(context: &Context<'_>) {
    let n = 6usize;
    let mut dft = tensor(context, vec![n, n], vec![SY, NS]);
    let mut idft = tensor(context, vec![n, n], vec![SY, NS]);
    let mut mesh = tensor(context, vec![n, n, n], vec![NS, NS, NS]);

    dft.transform(|key, value| {
        let angle = -2.0 * (key / n) as f64 * (key % n) as f64 * (std::f64::consts::PI / n as f64);
        let root = cis(angle);
        *value = Scalar::new(root.re / n as f64, root.im / n as f64);
    });
    idft.transform(|key, value| {
        let angle = 2.0 * (key / n) as f64 * (key % n) as f64 * (std::f64::consts::PI / n as f64);
        let root = cis(angle);
        *value = Scalar::new(root.re / n as f64, root.im / n as f64);
    });
    mesh.transform(|key, value| {
        let x = key % n;
        let y = (key / n) % n;
        let z = key / (n * n);
        let mut sum = Scalar::new(0.0, 0.0);
        for j in 0..n {
            let term = cis(-2.0 * std::f64::consts::PI * j as f64 / n as f64 * (x + y + z) as f64);
            sum.re += term.re;
            sum.im += term.im;
        }
        *value = sum;
    });

    let one = Scalar::new(1.0, 0.0);
    let zero = Scalar::new(0.0, 0.0);
    let mut first = tensor(context, vec![n, n, n], vec![NS, NS, NS]);
    first
        .contract_from("iqr", &dft, "ip", &mesh, "pqr", one, zero, true)
        .unwrap();
    let mut second = tensor(context, vec![n, n, n], vec![NS, NS, NS]);
    second
        .contract_from("ijr", &dft, "jq", &first, "iqr", one, zero, true)
        .unwrap();
    let mut transformed = tensor(context, vec![n, n, n], vec![NS, NS, NS]);
    transformed
        .contract_from("ijk", &dft, "kr", &second, "ijr", one, zero, true)
        .unwrap();

    let mut pass = 1i32;
    for (key, value) in transformed.local_pairs() {
        let i = key % n;
        let j = (key / n) % n;
        let k = key / (n * n);
        let expected = if i == j && i == k { 1.0 } else { 0.0 };
        if (value.re - expected).abs() >= 1e-9 {
            pass = 0;
        }
    }
    let minimum = CustomMonoid {
        identity: 1i32,
        addition: |left: &i32, right: &i32| (*left).min(*right),
    };
    context.all_reduce_monoid(&minimum, std::slice::from_mut(&mut pass), true);
    assert_eq!(pass, 1);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS dft_3D: 3D normalized DFT contraction produces the diagonal mesh; n=6; per-element real error<1e-9; world+parity"
        );
    }
    world.close();
    drop(universe);
}
