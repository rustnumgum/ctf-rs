// Migrated from pinned test/subworld_gemm.cxx; source Frobenius error < 1e-9.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    linalg::Native,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

// POSIX srand48/drand48 recurrence, avoiding a platform-specific libc dependency.
fn draw(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(0x5deece66d).wrapping_add(0xb) & ((1_u64 << 48) - 1);
    *state as f64 / (1_u64 << 48) as f64
}
fn case(c: &Context<'_>, m: usize, n: usize, k: usize, div: usize) {
    let div = div.min(c.size());
    let child_size = c.size() / div;
    let color = c.rank() / child_size;
    let child = c
        .split(Some(color as i32), (c.rank() % child_size) as i32)
        .unwrap();
    let da = Distribution::cyclic(vec![m, k], c.size());
    let db = Distribution::cyclic(vec![k, n], c.size());
    let dc = Distribution::cyclic(vec![m, n], c.size());
    let mut a = Tensor::new(c, da, Arithmetic::<f64>::new());
    let mut b = Tensor::new(c, db, Arithmetic::<f64>::new());
    let mut state = (((13 * c.rank()) as u64) << 16) | 0x330e;
    a.transform(|_, x| *x = draw(&mut state) - 0.5);
    b.transform(|_, x| *x = draw(&mut state) - 0.5);
    let mut answer = Tensor::new(c, dc.clone(), Arithmetic::<f64>::new());
    answer.gemm_2d::<Native>(&a, &b, [1, c.size()], div as f64, 0.0);
    let mut output = Tensor::new(c, dc, Arithmetic::<f64>::new());
    {
        let sub_da = Distribution::cyclic(vec![m, k], child_size);
        let sub_db = Distribution::cyclic(vec![k, n], child_size);
        let sub_dc = Distribution::cyclic(vec![m, n], child_size);
        let mut sub_a = Tensor::new(&child, sub_da.clone(), Arithmetic::<f64>::new());
        let mut sub_b = Tensor::new(&child, sub_db.clone(), Arithmetic::<f64>::new());
        let mut sub_c = Tensor::new(&child, sub_dc.clone(), Arithmetic::<f64>::new());
        for selected in 0..div {
            a.add_to_subworld(
                if selected == color {
                    Some(&mut sub_a)
                } else {
                    None
                },
                &sub_da,
                1.0,
                0.0,
            );
            b.add_to_subworld(
                if selected == color {
                    Some(&mut sub_b)
                } else {
                    None
                },
                &sub_db,
                1.0,
                0.0,
            );
        }
        sub_c.gemm_2d::<Native>(&sub_a, &sub_b, [1, child_size], 1.0, 0.0);
        for selected in 0..div {
            output.add_from_subworld(
                if selected == color {
                    Some(&sub_c)
                } else {
                    None
                },
                &sub_dc,
                1.0,
                1.0,
            );
        }
    }
    child.close();
    answer
        .sum_from(
            "ij",
            &output,
            "ij",
            Topology::new(vec![c.size()]),
            -1.0,
            1.0,
        )
        .unwrap();
    let error = answer.norm2();
    assert!(
        error.is_finite() && error < 1e-9,
        "source subworld GEMM residual {error}"
    );
}
fn run(c: &Context<'_>) {
    for div in [1, 2, 4] {
        case(c, 17, 23, 31, div);
        case(c, 2, 3, 1, div);
    }
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_subworld_gemm: source replicated-product identity, drand48 sequence, default/tiny shapes, divisions1/2/4, Frobenius<1e-9; world+parity"
        );
    }
    world.close();
    drop(universe);
}
