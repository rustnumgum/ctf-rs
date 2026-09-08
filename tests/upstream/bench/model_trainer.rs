//! Dense native counterpart of the pinned `bench/model_trainer.cxx`.
//!
//! The benchmark keeps the source driver's communicator split, dimension
//! progression, and five-round training structure.  The sparse MTTKRP,
//! sparse vector/matrix, offload, and sparse MP3 workloads are intentionally
//! outside this dense subset.  The source model registry updates are also
//! excluded because there is no native model-registry responsibility here;
//! communicator barriers retain the phase boundaries instead.

use std::time::Instant;

use ctf::{
    algebra::{Arithmetic, Monoid},
    context::Context,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

const TIME_BUDGET: f64 = 5.0;
const NUM_ITERATIONS: usize = 5;
const TIME_JUMP: f64 = 1.5;

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

#[derive(Clone, Copy)]
struct Maximum;

impl Monoid for Maximum {
    type Element = f64;

    fn zero(&self) -> Self::Element {
        0.0
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (*left).max(*right)
    }
}

fn distribution(context: &Context<'_>, shape: &[usize]) -> Distribution {
    Distribution::cyclic(shape.to_vec(), context.size())
}

fn fill_dense(tensor: &mut Dense<'_, '_>, salt: usize) {
    tensor.transform(|key, value| {
        *value = (1 + (key.wrapping_mul(17).wrapping_add(salt) % 23)) as f64 / 23.0;
    });
}

fn base_labels(order: usize) -> String {
    (0..order - 2)
        .map(|axis| (b'a' + axis as u8) as char)
        .collect()
}

/// Port the source's dense tensor-times-matrix chain for orders two through six.
fn train_ttm(size: usize, matrix_length: usize, context: &Context<'_>) {
    let topology = Topology::new(vec![context.size()]);
    for order in 2..7 {
        let mut n = 1usize;
        while n.pow(order as u32) < size {
            n += 1;
        }
        let tensor_shape = vec![n; order];

        let mut once_matrix_shape = vec![n; order];
        once_matrix_shape[order - 1] = matrix_length;
        let mut twice_matrix_shape = once_matrix_shape.clone();
        twice_matrix_shape[order - 2] = matrix_length;
        let twice_matrix_output_shape = twice_matrix_shape[..order - 1].to_vec();

        let mut tensor = Tensor::new(
            context,
            distribution(context, &tensor_shape),
            Arithmetic::<f64>::new(),
        );
        let mut once_matrix = Tensor::new(
            context,
            distribution(context, &once_matrix_shape),
            Arithmetic::<f64>::new(),
        );
        let mut twice_matrix = Tensor::new(
            context,
            distribution(context, &twice_matrix_shape),
            Arithmetic::<f64>::new(),
        );
        let mut matrix = Tensor::new(
            context,
            distribution(context, &[n, matrix_length]),
            Arithmetic::<f64>::new(),
        );
        fill_dense(&mut tensor, order);
        fill_dense(&mut matrix, order + 19);

        let base = base_labels(order);
        let first_input = format!("{base}zx");
        let first_output = format!("{base}zy");
        once_matrix
            .contract_from(
                &first_output,
                &tensor,
                &first_input,
                &matrix,
                "xy",
                topology.clone(),
                1.0,
                0.0,
            )
            .unwrap();

        let second_input = format!("{base}xq");
        let second_output = format!("{base}yq");
        twice_matrix
            .contract_from(
                &second_output,
                &once_matrix,
                &second_input,
                &matrix,
                "xy",
                topology.clone(),
                1.0,
                0.0,
            )
            .unwrap();

        let third_input = format!("{base}xy");
        let third_output = format!("{base}y");
        let mut final_matrix = Tensor::new(
            context,
            distribution(context, &twice_matrix_output_shape),
            Arithmetic::<f64>::new(),
        );
        final_matrix
            .contract_from(
                &third_output,
                &once_matrix,
                &third_input,
                &matrix,
                "xy",
                topology.clone(),
                1.0,
                0.0,
            )
            .unwrap();
    }
}

/// Port the source's dense vector/matrix contraction and custom-function chain.
fn train_dns_vec_mat(n: usize, m: usize, context: &Context<'_>) {
    let topology = Topology::new(vec![context.size()]);
    let mut b = Tensor::new(
        context,
        distribution(context, &[n]),
        Arithmetic::<f64>::new(),
    );
    let mut c = Tensor::new(
        context,
        distribution(context, &[m]),
        Arithmetic::<f64>::new(),
    );
    let mut a = Tensor::new(
        context,
        distribution(context, &[m, n]),
        Arithmetic::<f64>::new(),
    );
    let mut a1 = Tensor::new(
        context,
        distribution(context, &[m, n]),
        Arithmetic::<f64>::new(),
    );
    let mut a2 = Tensor::new(
        context,
        distribution(context, &[m, n]),
        Arithmetic::<f64>::new(),
    );
    let mut g = Tensor::new(
        context,
        distribution(context, &[n, n]),
        Arithmetic::<f64>::new(),
    );
    let mut f = Tensor::new(
        context,
        distribution(context, &[m, m]),
        Arithmetic::<f64>::new(),
    );
    fill_dense(&mut b, 1);
    fill_dense(&mut c, 2);
    fill_dense(&mut a, 3);
    fill_dense(&mut a1, 4);
    fill_dense(&mut a2, 5);
    fill_dense(&mut g, 6);
    fill_dense(&mut f, 7);

    let old_a = a.clone();
    a.contract_from("ij", &old_a, "ik", &g, "kj", topology.clone(), 1.0, 1.0)
        .unwrap();
    let old_a = a.clone();
    a.contract_from("ij", &old_a, "ij", &a1, "ij", topology.clone(), 1.0, 1.0)
        .unwrap();
    let old_a = a.clone();
    a.contract_from("ij", &f, "ik", &old_a, "kj", topology.clone(), 1.0, 1.0)
        .unwrap();
    c.contract_from("i", &a, "ij", &b, "j", topology.clone(), 1.0, 1.0)
        .unwrap();
    b.contract_from("j", &a, "ij", &c, "i", topology.clone(), 0.2, 1.0)
        .unwrap();
    let old_b = b.clone();
    b.contract_from("i", &old_b, "i", &old_b, "i", topology.clone(), 1.0, 1.0)
        .unwrap();

    a2.sum_function_from("ij", &a, "ij", topology.clone(), 1.0, 0.0, |value| {
        value * value
    })
    .unwrap();
    c.sum_function_from("i", &a, "ij", topology.clone(), 1.0, 1.0, |value| {
        value * value
    })
    .unwrap();
    let old_a = a.clone();
    a1.contract_function_on(
        "ij",
        &old_a,
        "kj",
        &f,
        "ki",
        topology.clone(),
        "i",
        -1.0,
        1.0,
        |left, right| left * left + right * right,
    );

    b.transform(|_, value| *value *= *value);
    a.transform(|_, value| *value *= *value);
    b.transform(|_, value| *value -= *value / *value);
    let a_values = a.local_pairs();
    let mut position = 0;
    a2.transform(|key, value| {
        let (a_key, a_value) = a_values[position];
        debug_assert_eq!(a_key, key);
        *value -= *value / a_value;
        position += 1;
    });
}

fn train_world(duration: f64, context: &Context<'_>, step_size: f64) {
    let mut n = 19usize;
    let m0 = 75usize;
    let approximate_iterations = (step_size * step_size * 10.0 * duration.ln()).max(1.0) as usize;
    let target_duration = duration / approximate_iterations as f64;
    let maximum = Maximum;

    loop {
        let started = Instant::now();
        let mut iterations = 0usize;
        let mut m = m0;
        let mut elapsed = 0.0;
        loop {
            if n < 80 {
                train_ttm(n * m + 13, n, context);
            }
            train_dns_vec_mat(n, m, context);
            iterations += 1;
            m = (m as f64 * step_size) as usize;
            n += 2;
            elapsed = started.elapsed().as_secs_f64();
            context.all_reduce_monoid(&maximum, std::slice::from_mut(&mut elapsed), true);
            if !(elapsed < target_duration && m <= 1_000_000) {
                break;
            }
        }
        if iterations <= 2 || n >= 1_000_000 {
            break;
        }
        n = (n as f64 * step_size) as usize;
        m += 3;
    }
}

fn floor_log2(value: usize) -> usize {
    assert!(value > 0);
    (usize::BITS - 1 - value.leading_zeros()) as usize
}

fn train_all(context: &Context<'_>, time: f64, num_iterations: usize, time_jump: f64) {
    assert!(time > 0.0);
    assert!(num_iterations > 0);
    assert!(time_jump > 1.0);

    let rank = context.rank();
    let color = floor_log2(rank + 1);
    let end_color = floor_log2(context.size() + 1);
    let key = rank + 1 - (1usize << color);
    let child = context.split(Some(color as i32), key as i32).unwrap();
    let mut dtime = (time / (1.0 - 1.0 / time_jump)) / time_jump.powf(num_iterations as f64 - 1.0);

    for iteration in 0..num_iterations {
        let step_size = 1.0 + 1.5 / 2.0f64.powi(iteration as i32);
        if color != end_color {
            train_world(dtime / 5.0, &child, step_size);
            // The source updates its global model registry here; that
            // interface is intentionally excluded from this dense port.
            child.barrier();
        }

        if color != end_color {
            train_world(dtime / 5.0, &child, step_size);
        }
        context.barrier();

        if color != end_color {
            train_world(dtime / 5.0, &child, step_size);
            child.barrier();
        }

        if color != end_color {
            train_world(dtime / 5.0, &child, step_size);
        }
        context.barrier();

        train_world(dtime / 5.0, context, step_size);
        context.barrier();
        dtime *= time_jump;
    }
    child.close();
}

fn run(context: &Context<'_>) -> f64 {
    context.barrier();
    let started = Instant::now();
    train_all(context, TIME_BUDGET, NUM_ITERATIONS, TIME_JUMP);
    let elapsed = started.elapsed().as_secs_f64();
    context.barrier();
    elapsed
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let world_seconds = run(&world);
    if world.rank() == 0 {
        println!(
            "INFO model_trainer: dense train_ttm+train_dns_vec_mat, time={TIME_BUDGET}, iterations={NUM_ITERATIONS}, time_jump={TIME_JUMP}, world sec={world_seconds}"
        );
    }
    world.close();
    drop(universe);
}
