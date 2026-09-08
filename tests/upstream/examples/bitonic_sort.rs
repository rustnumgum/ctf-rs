//! Pinned `examples/bitonic_sort.cxx`: iterative bitonic sort through
//! min-semiring tensor contractions.

use ctf::{
    algebra::{CustomMonoid, CustomSemiring},
    context::Context,
    mapping::{Distribution, Topology},
    random::Generator,
    tensor::Tensor,
};

type MinAlgebra = CustomSemiring<CustomMonoid<f64, fn(&f64, &f64) -> f64>, fn(&f64, &f64) -> f64>;
type MinTensor<'c, 'r> = Tensor<'c, 'r, MinAlgebra>;

fn minimum(left: &f64, right: &f64) -> f64 {
    (*left).min(*right)
}

fn product(left: &f64, right: &f64) -> f64 {
    *left * *right
}

fn min_algebra() -> MinAlgebra {
    CustomSemiring {
        monoid: CustomMonoid {
            identity: f64::MAX / 2.0,
            addition: minimum,
        },
        identity: 1.0,
        multiplication: product,
    }
}

fn labels(logn: usize) -> String {
    (0..logn).map(|axis| (b'a' + axis as u8) as char).collect()
}

fn tensor<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    algebra: &MinAlgebra,
) -> MinTensor<'c, 'r> {
    Tensor::new(
        context,
        Distribution::cyclic(shape, context.size()),
        algebra.clone(),
    )
}

fn bitonic_sort(context: &Context<'_>, logn: usize) {
    let length = 1usize << logn;
    let mut vector = Tensor::new(
        context,
        Distribution::cyclic(vec![length], context.size()),
        ctf::algebra::Arithmetic::<f64>::new(),
    );
    let mut generator = Generator::new(27 * context.rank() as u64);
    vector.fill_random(0.0, 1.0, &mut generator);

    let algebra = min_algebra();
    let mut values = tensor(context, vec![2; logn], &algebra);
    values.write_add(&vector.local_pairs());

    let mut swap_up_down = tensor(context, vec![2, 2, 2], &algebra);
    if context.rank() == 0 {
        swap_up_down.write_add(&[
            (0, 1.0),
            (1, 1.0),
            (2, -1.0),
            (3, -1.0),
            (4, -1.0),
            (5, -1.0),
            (6, 1.0),
            (7, 1.0),
        ]);
    } else {
        swap_up_down.write_add(&[]);
    }
    let mut fix_sign_up_down = tensor(context, vec![2, 2], &algebra);
    if context.rank() == 0 {
        fix_sign_up_down.write_add(&[(0, 1.0), (1, -1.0), (2, -1.0), (3, 1.0)]);
    } else {
        fix_sign_up_down.write_add(&[]);
    }
    let mut swap_up = tensor(context, vec![2, 2], &algebra);
    if context.rank() == 0 {
        swap_up.write_add(&[(0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0)]);
    } else {
        swap_up.write_add(&[]);
    }
    let mut fix_sign_up = tensor(context, vec![2], &algebra);
    if context.rank() == 0 {
        fix_sign_up.write_add(&[(0, 1.0), (1, -1.0)]);
    } else {
        fix_sign_up.write_add(&[]);
    }

    let index = labels(logn);
    let topology = Topology::new(vec![context.size()]);
    let mut index_z = index.clone().into_bytes();
    for level in 0..logn - 1 {
        let up_down = (b'a' + (level + 1) as u8) as char;
        for axis in (0..=level).rev() {
            index_z[axis] = b'z';
            let swap_labels = format!("z{}{}", index.as_bytes()[axis] as char, up_down);
            let fix_labels = format!("{}{}", index.as_bytes()[axis] as char, up_down);
            let old = values.clone();
            values
                .contract_from(
                    &index,
                    &swap_up_down,
                    &swap_labels,
                    &old,
                    &String::from_utf8(index_z.clone()).unwrap(),
                    topology.clone(),
                    1.0,
                    f64::MAX / 2.0,
                )
                .unwrap();
            let old = values.clone();
            values
                .contract_from(
                    &index,
                    &fix_sign_up_down,
                    &fix_labels,
                    &old,
                    &index,
                    topology.clone(),
                    1.0,
                    f64::MAX / 2.0,
                )
                .unwrap();
            index_z[axis] = index.as_bytes()[axis];
        }
    }
    for axis in (0..logn).rev() {
        index_z[axis] = b'z';
        let swap_labels = format!("z{}", index.as_bytes()[axis] as char);
        let old = values.clone();
        values
            .contract_from(
                &index,
                &swap_up,
                &swap_labels,
                &old,
                &String::from_utf8(index_z.clone()).unwrap(),
                topology.clone(),
                1.0,
                f64::MAX / 2.0,
            )
            .unwrap();
        let old = values.clone();
        let fix_labels = (index.as_bytes()[axis] as char).to_string();
        values
            .contract_from(
                &index,
                &fix_sign_up,
                &fix_labels,
                &old,
                &index,
                topology.clone(),
                1.0,
                f64::MAX / 2.0,
            )
            .unwrap();
        index_z[axis] = index.as_bytes()[axis];
    }

    let rank = context.rank();
    let pairs: Vec<_> = values
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| values.distribution().owner(*key) == rank)
        .collect();
    vector.write_scaled(&pairs, &1.0, &0.0);

    let mut keys = Vec::new();
    for key in 1..length {
        if vector.distribution().owner(key) == rank {
            keys.push(key - 1);
            keys.push(key);
        }
    }
    let data = vector.read(&keys);
    let mut pass = 1i32;
    for pair in data.chunks_exact(2) {
        if pair[1] < pair[0] {
            pass = 0;
        }
    }
    let minimum_pass = CustomMonoid {
        identity: 1i32,
        addition: |left: &i32, right: &i32| (*left).min(*right),
    };
    context.all_reduce_monoid(&minimum_pass, std::slice::from_mut(&mut pass), true);
    assert_eq!(pass, 1);
}

fn run(context: &Context<'_>) {
    bitonic_sort(context, 4);
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
            "DIGIT / PASS bitonic_sort: min-semiring tensor contractions sort the random vector exactly; logn=4; exact nondecreasing order; world+parity"
        );
    }
    world.close();
    drop(universe);
}
