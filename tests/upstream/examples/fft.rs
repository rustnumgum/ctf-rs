//! Pinned examples/fft.cxx: iterative radix-2 FFT using tensor products.

use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Semiring},
    context::Context,
    mapping::{Distribution, Topology},
    random::Generator,
    tensor::Tensor,
    util::factorize,
};

type Scalar = Complex<f64>;
type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<Scalar>>;

fn omega(i: usize, n: usize) -> Scalar {
    let angle = -2.0 * i as f64 * std::f64::consts::PI / n as f64;
    Scalar::new(angle.cos(), angle.sin())
}

fn distribution(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    Distribution::cyclic(shape, context.size())
}

fn dense<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>) -> Dense<'c, 'r> {
    Dense::new(context, distribution(context, shape), Arithmetic::new())
}

fn dft_matrix<'c, 'r>(context: &'c Context<'r>, n: usize) -> Dense<'c, 'r> {
    let mut dft = dense(context, vec![n, n]);
    dft.transform(|key, value| *value = omega((key / n) * (key % n), n));
    dft
}

fn twiddle_matrix<'c, 'r>(context: &'c Context<'r>, n: usize) -> Dense<'c, 'r> {
    let mut twiddle = dense(context, vec![2, 2]);
    twiddle.transform(|key, value| {
        *value = if key < 3 {
            Scalar::new(1.0, 0.0)
        } else {
            omega(1, n)
        };
    });
    twiddle
}

fn labels(order: usize) -> String {
    assert!(order <= 25);
    String::from_utf8((0..order).map(|axis| b'a' + axis as u8).collect()).unwrap()
}

fn fft<'c, 'r>(vector: &Dense<'c, 'r>, n: usize) -> Dense<'c, 'r> {
    let mut log_n = 0usize;
    while 1usize << log_n < n {
        log_n += 1;
    }
    assert_eq!(1usize << log_n, n);

    let factors = factorize(n as i64);
    assert_eq!(factors.len(), log_n);
    assert!(factors.iter().all(|&factor| factor == 2));
    let shape: Vec<_> = factors.into_iter().map(|factor| factor as usize).collect();
    let full_labels = labels(log_n);
    let reverse_labels: String = full_labels.chars().rev().collect();
    let topology = Topology::new(vec![vector.context().size()]);

    let folded = vector.reshape(distribution(vector.context(), shape.clone()));
    let mut values = dense(vector.context(), shape.clone());
    values
        .sum_from(
            &reverse_labels,
            &folded,
            &full_labels,
            topology.clone(),
            Scalar::new(1.0, 0.0),
            Scalar::new(0.0, 0.0),
        )
        .unwrap();

    let dft = dft_matrix(vector.context(), 2);
    let mut s_old = dense(vector.context(), vec![2]);
    s_old.transform(|_, value| *value = Scalar::new(1.0, 0.0));
    let mut m = 1usize;

    for stage in 0..log_n {
        m *= 2;
        let prefix = &full_labels[..=stage];
        let s = if stage == 0 {
            s_old.clone()
        } else {
            let mut current = dense(vector.context(), vec![2; stage + 1]);
            let twiddle = twiddle_matrix(vector.context(), m);
            let twiddle_labels = format!("a{}", full_labels.as_bytes()[stage] as char);
            current
                .contract_from(
                    prefix,
                    &twiddle,
                    &twiddle_labels,
                    &s_old,
                    &prefix[1..],
                    topology.clone(),
                    Scalar::new(1.0, 0.0),
                    Scalar::new(0.0, 0.0),
                )
                .unwrap();
            current
        };
        s_old = s.clone();

        let mut scaled = dense(vector.context(), shape.clone());
        scaled
            .contract_from(
                &full_labels,
                &values,
                &full_labels,
                &s,
                prefix,
                topology.clone(),
                Scalar::new(1.0, 0.0),
                Scalar::new(0.0, 0.0),
            )
            .unwrap();

        let mut input_labels = full_labels.clone().into_bytes();
        input_labels[stage] = b'z';
        let input_labels = String::from_utf8(input_labels).unwrap();
        let dft_labels = format!("{}z", full_labels.as_bytes()[stage] as char);
        let mut next = dense(vector.context(), shape.clone());
        next.contract_from(
            &full_labels,
            &dft,
            &dft_labels,
            &scaled,
            &input_labels,
            topology.clone(),
            Scalar::new(1.0, 0.0),
            Scalar::new(0.0, 0.0),
        )
        .unwrap();
        values = next;
    }

    values.reshape(vector.distribution().clone())
}

fn run(context: &Context<'_>) {
    let n = 16usize;
    let mut a = dense(context, vec![n]);
    let mut generator = Generator::new((2 * context.rank()) as u64);
    a.transform(|_, value| {
        *value = Scalar::new(generator.unit_interval(), generator.unit_interval());
    });

    let dft = dft_matrix(context, n);
    let mut direct = dense(context, vec![n]);
    direct
        .contract_from(
            "i",
            &dft,
            "ij",
            &a,
            "j",
            Topology::new(vec![context.size()]),
            Scalar::new(1.0, 0.0),
            Scalar::new(0.0, 0.0),
        )
        .unwrap();
    let transformed = fft(&a, n);
    direct
        .sum_from(
            "i",
            &transformed,
            "i",
            Topology::new(vec![context.size()]),
            Scalar::new(-1.0, 0.0),
            Scalar::new(1.0, 0.0),
        )
        .unwrap();

    let algebra = Arithmetic::<Scalar>::new();
    let mut norm = algebra.zero();
    for (key, value) in direct.local_pairs() {
        if direct.distribution().owner(key) == context.rank() {
            norm = algebra.add(&norm, &algebra.multiply(&value, &value));
        }
    }
    context.all_reduce_monoid(&algebra, std::slice::from_mut(&mut norm), true);
    assert!(norm.re.abs() <= 1e-6, "FFT real residual {}", norm.re);
    assert!(norm.im.abs() <= 1e-6, "FFT imaginary residual {}", norm.im);
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
            "DIGIT / PASS fft: iterative folded radix-2 FFT equals DFT; n=16; real/imag residual<=1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
