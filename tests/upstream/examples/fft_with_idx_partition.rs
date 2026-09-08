// Pinned examples/fft_with_idx_partition.cxx: fiber FFT on a user partition.

use ctf::{
    algebra::{Arithmetic, Complex},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    partition::Partition,
    random::Generator,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{NS, SY},
};

type Scalar = Complex<f64>;
type Tensor<'c, 'r> = SymmetricTensor<'c, 'r, Arithmetic<Scalar>>;

fn cis(angle: f64) -> Scalar {
    Scalar::new(angle.cos(), angle.sin())
}

fn multiply(a: Scalar, b: Scalar) -> Scalar {
    Scalar::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn fft(input: &[Scalar], output: &mut [Scalar]) {
    let m = input.len();
    assert_eq!(output.len(), m);
    if m == 1 {
        output[0] = input[0];
        return;
    }
    assert_eq!(m % 2, 0);

    let mut even = Vec::with_capacity(m / 2);
    let mut odd = Vec::with_capacity(m / 2);
    for pair in input.chunks_exact(2) {
        even.push(pair[0]);
        odd.push(pair[1]);
    }
    let mut u = vec![Scalar::default(); m / 2];
    let mut v = vec![Scalar::default(); m / 2];
    fft(&even, &mut u);
    fft(&odd, &mut v);
    for i in 0..m / 2 {
        let z = multiply(cis(-2.0 * i as f64 * std::f64::consts::PI / m as f64), v[i]);
        output[i] = Scalar::new(u[i].re + z.re, u[i].im + z.im);
        output[m / 2 + i] = Scalar::new(u[i].re - z.re, u[i].im - z.im);
    }
}

fn default_distribution(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !shape.is_empty() {
        mappings[0].augment_physical(&topology, 0);
    }
    Distribution::new(shape, topology, mappings)
}

fn tensor<'c, 'r>(
    context: &'c Context<'r>,
    mut distribution: Distribution,
    links: Vec<ctf::symmetry::Symmetry>,
) -> Tensor<'c, 'r> {
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
    let log_m = 8usize;
    let m = 1usize << log_m;

    let mut dft = tensor(
        context,
        default_distribution(context, vec![m, m]),
        vec![SY, NS],
    );
    dft.transform(|key, value| {
        let angle = -2.0 * (key / m) as f64 * (key % m) as f64 * (std::f64::consts::PI / m as f64);
        *value = cis(angle);
    });

    let mut pr = (context.size() as f64).sqrt() as usize;
    let mut pc = context.size() / pr;
    while pr * pc != context.size() {
        pr += 1;
        pc = context.size() / pr;
    }
    let partition = Partition::new([pr, pc]);
    let indexed = partition.indexed("jk");
    let distribution = indexed.distribution(vec![m, n, n], "ijk");
    let mut a = tensor(context, distribution.clone(), vec![NS, NS, NS]);
    let mut b = tensor(context, distribution, vec![NS, NS, NS]);
    let mut c = tensor(
        context,
        default_distribution(context, vec![m, n, n]),
        vec![NS, NS, NS],
    );

    let mut generator = Generator::new(context.rank() as u64);
    a.fill_random(Scalar::new(0.0, 0.0), Scalar::new(1.0, 1.0), &mut generator);
    c.contract_from(
        "ijk",
        &dft,
        "il",
        &a,
        "ljk",
        Scalar::new(1.0, 0.0),
        Scalar::new(0.0, 0.0),
        true,
    )
    .unwrap();

    let raw_a = a.local_storage();
    assert_eq!(raw_a.len(), b.local_storage().len());
    assert_eq!(raw_a.len() % m, 0);
    let mut raw_b = vec![Scalar::default(); raw_a.len()];
    for (input, output) in raw_a.chunks_exact(m).zip(raw_b.chunks_exact_mut(m)) {
        fft(input, output);
    }
    let b_distribution = b.distribution().clone();
    let rank = context.rank();
    b.transform(|key, value| {
        *value = raw_b[b_distribution.local_offset(rank, key)];
    });

    c.sum_from(
        "ijk",
        &b,
        "ijk",
        Scalar::new(-1.0, 0.0),
        Scalar::new(1.0, 0.0),
    );
    let error = c.norm2();
    assert!(
        error <= (n * n * m) as f64 * 1e-6,
        "FFT partition error {error}"
    );
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
            "DIGIT / PASS fft_with_idx_partition: recursive fiber FFT equals SY DFT contraction; n=6 logm=8; error<=n*n*m*1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
