// Pinned `examples/recursive_matmul.cxx`: recursive GEMM using slices and
// communicator-local subworlds.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    random::Generator,
    tensor::Tensor,
};

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

fn recursive_matmul<'c, 'r>(
    n: usize,
    m: usize,
    k: usize,
    a: &Dense<'c, 'r>,
    b: &Dense<'c, 'r>,
    c: &mut Dense<'c, 'r>,
) {
    let processes = c.context().size();
    if processes == 1 || m == 1 || n == 1 || k == 1 {
        c.contract_from(
            "ij",
            a,
            "ik",
            b,
            "kj",
            Topology::new(vec![processes]),
            1.0,
            1.0,
        )
        .unwrap();
        return;
    }

    let mut divisor = 2;
    while processes % divisor != 0 {
        divisor += 1;
    }
    let child_size = processes / divisor;
    let rank = c.context().rank();
    let group = rank / child_size;
    let child = c
        .context()
        .split(Some(group as i32), (rank % child_size) as i32)
        .unwrap();

    let (ni, nj, nk) = if m >= n && m >= k {
        assert_eq!(m % divisor, 0);
        (divisor, 1, 1)
    } else if n >= m && n >= k {
        assert_eq!(n % divisor, 0);
        (1, divisor, 1)
    } else {
        assert_eq!(k % divisor, 0);
        (1, 1, divisor)
    };
    {
        let child_a_distribution = Distribution::cyclic(vec![m / ni, k / nk], child_size);
        let child_b_distribution = Distribution::cyclic(vec![k / nk, n / nj], child_size);
        let child_c_distribution = Distribution::cyclic(vec![m / ni, n / nj], child_size);
        let mut child_a = Dense::new(&child, child_a_distribution.clone(), Arithmetic::new());
        let mut child_b = Dense::new(&child, child_b_distribution.clone(), Arithmetic::new());
        for selected in 0..divisor {
            let (ri, rj, rk) = if ni > 1 {
                (selected, 0, 0)
            } else if nj > 1 {
                (0, selected, 0)
            } else {
                (0, 0, selected)
            };
            let off_ik = [ri * m / ni, rk * k / nk];
            let off_kj = [rk * k / nk, rj * n / nj];
            let a_slice = a.slice(&[off_ik[0]..off_ik[0] + m / ni, off_ik[1]..off_ik[1] + k / nk]);
            let b_slice = b.slice(&[off_kj[0]..off_kj[0] + k / nk, off_kj[1]..off_kj[1] + n / nj]);
            a_slice.add_to_subworld(
                (selected == group).then_some(&mut child_a),
                &child_a_distribution,
                1.0,
                0.0,
            );
            b_slice.add_to_subworld(
                (selected == group).then_some(&mut child_b),
                &child_b_distribution,
                1.0,
                0.0,
            );
        }

        let mut child_c = Dense::new(&child, child_c_distribution.clone(), Arithmetic::new());
        recursive_matmul(n / nj, m / ni, k / nk, &child_a, &child_b, &mut child_c);
        let source_ranges = [0..m / ni, 0..n / nj];
        for selected in 0..divisor {
            let (ri, rj) = if ni > 1 {
                (selected, 0)
            } else if nj > 1 {
                (0, selected)
            } else {
                (0, 0)
            };
            let mut parent_block = Dense::new(
                c.context(),
                Distribution::cyclic(vec![m / ni, n / nj], processes),
                Arithmetic::new(),
            );
            parent_block.add_from_subworld(
                (selected == group).then_some(&child_c),
                &child_c_distribution,
                1.0,
                0.0,
            );
            c.assign_slice(
                &[
                    ri * m / ni..(ri + 1) * m / ni,
                    rj * n / nj..(rj + 1) * n / nj,
                ],
                &parent_block,
                &source_ranges,
                &1.0,
                &1.0,
            );
        }
    }
    child.close();
}

fn case(context: &Context<'_>, n: usize, m: usize, k: usize) {
    let distribution = |shape| Distribution::cyclic(shape, context.size());
    let mut a = Dense::new(context, distribution(vec![m, k]), Arithmetic::new());
    let mut b = Dense::new(context, distribution(vec![k, n]), Arithmetic::new());
    let mut c = Dense::new(context, distribution(vec![m, n]), Arithmetic::new());
    let mut generator = Generator::new(13 * context.rank() as u64);
    a.fill_random(-0.5, 0.5, &mut generator);
    b.fill_random(-0.5, 0.5, &mut generator);

    let mut reference = Dense::new(context, distribution(vec![m, n]), Arithmetic::new());
    reference
        .contract_from(
            "ij",
            &a,
            "ik",
            &b,
            "kj",
            Topology::new(vec![context.size()]),
            1.0,
            1.0,
        )
        .unwrap();
    recursive_matmul(n, m, k, &a, &b, &mut c);
    reference
        .sum_from(
            "ij",
            &c,
            "ij",
            Topology::new(vec![context.size()]),
            -1.0,
            1.0,
        )
        .unwrap();
    let error = reference.norm2();
    assert!(error < 1.0e-9, "recursive matmul residual {error:e}");
}

fn run(context: &Context<'_>) {
    case(context, 256, 128, 512);
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS recursive_matmul: recursive slice/subworld GEMM; n=256 m=128 k=512; residual<1e-9; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
