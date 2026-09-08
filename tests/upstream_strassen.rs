//! Bounded port of the active one-level Strassen driver in
//! `examples/strassen.cxx`.
//!
//! The pinned example does not recurse: it forms the seven half-size products
//! once, either on the parent communicator or on seven equal subworlds, and
//! then checks the sliced result against the ordinary product.  This test keeps
//! both source branches and uses dense slices only; expansion of the four
//! compressed input symmetries is an explicit input adaptation, not a gathered
//! computation.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, AS, NS, SH, SY},
    tensor::Tensor,
};

const N: usize = 8;
const H: usize = N / 2;
type Algebra = Arithmetic<f64>;
type Dense<'c, 'r> = Tensor<'c, 'r, Algebra>;

fn dense<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>) -> Dense<'c, 'r> {
    Tensor::new(
        context,
        Distribution::cyclic(shape, context.size()),
        Algebra::new(),
    )
}

fn symmetric_distribution(context: &Context<'_>, link: Symmetry) -> SymmetricDistribution {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; 2];
    mappings[0].augment_physical(&topology, 0);
    // Match the source's distributed compressed matrix shape while giving the
    // two axes equal total phases for SY/AS/SH.
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricDistribution::new(
        Distribution::new(vec![N, N], topology, mappings),
        vec![link, NS],
    )
}

fn inputs<'c, 'r>(context: &'c Context<'r>, link: Symmetry) -> (Dense<'c, 'r>, Dense<'c, 'r>) {
    let distribution = symmetric_distribution(context, link);
    let mut a = SymmetricTensor::new(context, distribution.clone(), Algebra::new());
    let mut b = SymmetricTensor::new(context, distribution, Algebra::new());
    // The source uses srand48(13*rank) and continues the stream from A into B.
    // MT19937-64 with this explicit seed is a bounded RNG adaptation, not
    // a claim of an identical drand48 stream.
    let mut generator = Generator::new(13 * context.rank() as u64);
    a.fill_random(-0.5, 0.5, &mut generator);
    b.fill_random(-0.5, 0.5, &mut generator);
    let target = Distribution::cyclic(vec![N, N], context.size());
    (a.unpack(target.clone()), b.unpack(target))
}

fn add<'c, 'r>(
    left: &Dense<'c, 'r>,
    right: &Dense<'c, 'r>,
    alpha: f64,
    topology: &Topology,
) -> Dense<'c, 'r> {
    let mut result = left.clone();
    result
        .sum_from("ij", right, "ij", topology.clone(), alpha, 1.0)
        .unwrap();
    result
}

fn multiply<'c, 'r>(
    left: &Dense<'c, 'r>,
    right: &Dense<'c, 'r>,
    topology: &Topology,
) -> Dense<'c, 'r> {
    let shape = vec![left.distribution().shape[0], right.distribution().shape[1]];
    let mut result = dense(left.context(), shape);
    result
        .contract_from("ij", left, "ik", right, "kj", topology.clone(), 1.0, 0.0)
        .unwrap();
    result
}

fn put<'c, 'r>(
    destination: &mut Dense<'c, 'r>,
    row: std::ops::Range<usize>,
    column: std::ops::Range<usize>,
    source: &Dense<'c, 'r>,
    alpha: f64,
    beta: f64,
) {
    let shape = source.distribution().shape.clone();
    destination.assign_slice(
        &[row, column],
        source,
        &[0..shape[0], 0..shape[1]],
        &alpha,
        &beta,
    );
}

fn transposed_block<'c, 'r>(
    context: &'c Context<'r>,
    block: &Dense<'c, 'r>,
    sign: f64,
    topology: &Topology,
) -> Dense<'c, 'r> {
    let transposed = block.permute_axes(&[1, 0]);
    let mut result = dense(context, vec![H, H]);
    result
        .sum_from("ij", &transposed, "ij", topology.clone(), sign, 0.0)
        .unwrap();
    result
}

fn strassen_parent<'c, 'r>(
    context: &'c Context<'r>,
    a: &Dense<'c, 'r>,
    b: &Dense<'c, 'r>,
    link: Symmetry,
) -> Dense<'c, 'r> {
    let topology = Topology::new(vec![context.size()]);
    let a11 = a.slice(&[0..H, 0..H]);
    let a21 = a.slice(&[H..N, 0..H]);
    let a22 = a.slice(&[H..N, H..N]);
    let a12 = match link {
        SY | SH => transposed_block(context, &a21, 1.0, &topology),
        AS => transposed_block(context, &a21, -1.0, &topology),
        NS => a.slice(&[0..H, H..N]),
    };
    let b11 = b.slice(&[0..H, 0..H]);
    let b21 = b.slice(&[H..N, 0..H]);
    let b22 = b.slice(&[H..N, H..N]);
    let b12 = match link {
        SY | SH => transposed_block(context, &b21, 1.0, &topology),
        AS => transposed_block(context, &b21, -1.0, &topology),
        NS => b.slice(&[0..H, H..N]),
    };

    // These are the seven active source equations, in the source's order.
    let m1 = multiply(
        &add(&a11, &a22, 1.0, &topology),
        &add(&b22, &b11, 1.0, &topology),
        &topology,
    );
    let m6 = multiply(
        &add(&a21, &a11, -1.0, &topology),
        &add(&b11, &b12, 1.0, &topology),
        &topology,
    );
    let m7 = multiply(
        &add(&a12, &a22, -1.0, &topology),
        &add(&b22, &b21, 1.0, &topology),
        &topology,
    );
    let m2 = multiply(&add(&a21, &a22, 1.0, &topology), &b11, &topology);
    let m5 = multiply(&add(&a11, &a12, 1.0, &topology), &b22, &topology);
    let m3 = multiply(&a11, &add(&b12, &b22, -1.0, &topology), &topology);
    let m4 = multiply(&a22, &add(&b21, &b11, -1.0, &topology), &topology);

    let mut result = dense(context, vec![N, N]);
    put(&mut result, 0..H, 0..H, &m1, 1.0, 0.0);
    put(&mut result, 0..H, 0..H, &m4, 1.0, 1.0);
    put(&mut result, 0..H, 0..H, &m5, -1.0, 1.0);
    put(&mut result, 0..H, 0..H, &m7, 1.0, 1.0);
    put(&mut result, 0..H, H..N, &m3, 1.0, 0.0);
    put(&mut result, 0..H, H..N, &m5, 1.0, 1.0);
    put(&mut result, H..N, 0..H, &m2, 1.0, 0.0);
    put(&mut result, H..N, 0..H, &m4, 1.0, 1.0);
    put(&mut result, H..N, H..N, &m1, 1.0, 0.0);
    put(&mut result, H..N, H..N, &m2, -1.0, 1.0);
    put(&mut result, H..N, H..N, &m3, 1.0, 1.0);
    put(&mut result, H..N, H..N, &m6, 1.0, 1.0);
    result
}

fn strassen_subworld<'c, 'r>(
    context: &'c Context<'r>,
    a: &Dense<'c, 'r>,
    b: &Dense<'c, 'r>,
) -> Dense<'c, 'r> {
    let child_size = context.size() / 7;
    let color = context.rank() / child_size;
    let child = context
        .split(Some(color as i32), (context.rank() % child_size) as i32)
        .unwrap();
    let child_topology = Topology::new(vec![child_size]);
    let parent_a11 = a.slice(&[0..H, 0..H]);
    let parent_a12 = a.slice(&[0..H, H..N]);
    let parent_a21 = a.slice(&[H..N, 0..H]);
    let parent_a22 = a.slice(&[H..N, H..N]);
    let parent_b11 = b.slice(&[0..H, 0..H]);
    let parent_b12 = b.slice(&[0..H, H..N]);
    let parent_b21 = b.slice(&[H..N, 0..H]);
    let parent_b22 = b.slice(&[H..N, H..N]);
    let child_distribution = Distribution::cyclic(vec![H, H], child_size);
    let mut result = dense(context, vec![N, N]);

    let mut child_a = Dense::new(&child, child_distribution.clone(), Algebra::new());
    let mut child_b = Dense::new(&child, child_distribution.clone(), Algebra::new());
    for selected in 0..7 {
        // Transfer only the quadrants used by this source product.  Each
        // transfer is collective on the parent, with None outside its group.
        match selected {
            0 => {
                parent_a11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_a22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
            }
            1 => {
                parent_a21.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_a11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    -1.0,
                    1.0,
                );
                parent_b11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b12.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
            }
            2 => {
                parent_a12.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_a22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    -1.0,
                    1.0,
                );
                parent_b22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b21.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
            }
            3 => {
                parent_a21.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_a22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
            }
            4 => {
                parent_a11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_a12.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
            }
            5 => {
                parent_a11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b12.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    -1.0,
                    1.0,
                );
            }
            6 => {
                parent_a22.add_to_subworld(
                    if selected == color {
                        Some(&mut child_a)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b21.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    1.0,
                    1.0,
                );
                parent_b11.add_to_subworld(
                    if selected == color {
                        Some(&mut child_b)
                    } else {
                        None
                    },
                    &child_distribution,
                    -1.0,
                    1.0,
                );
            }
            _ => unreachable!(),
        }
    }
    // All seven groups execute their products independently, as in source.
    // Parent transfers are explicitly serialized by the Rust subworld API.
    let product = multiply(&child_a, &child_b, &child_topology);
    for selected in 0..7 {
        let mut parent_product = dense(context, vec![H, H]);
        parent_product.add_from_subworld(
            if selected == color {
                Some(&product)
            } else {
                None
            },
            &child_distribution,
            1.0,
            1.0,
        );
        match selected {
            0 => {
                put(&mut result, 0..H, 0..H, &parent_product, 1.0, 0.0);
                put(&mut result, H..N, H..N, &parent_product, 1.0, 0.0);
            }
            1 => put(&mut result, H..N, H..N, &parent_product, 1.0, 1.0),
            2 => put(&mut result, 0..H, 0..H, &parent_product, 1.0, 1.0),
            3 => {
                put(&mut result, H..N, 0..H, &parent_product, 1.0, 0.0);
                put(&mut result, H..N, H..N, &parent_product, -1.0, 1.0);
            }
            4 => {
                put(&mut result, 0..H, 0..H, &parent_product, -1.0, 1.0);
                put(&mut result, 0..H, H..N, &parent_product, 1.0, 0.0);
            }
            5 => {
                put(&mut result, 0..H, H..N, &parent_product, 1.0, 1.0);
                put(&mut result, H..N, H..N, &parent_product, 1.0, 1.0);
            }
            6 => {
                put(&mut result, H..N, 0..H, &parent_product, 1.0, 1.0);
                put(&mut result, 0..H, 0..H, &parent_product, 1.0, 1.0);
            }
            _ => unreachable!(),
        }
    }
    child.close();
    result
}

fn strassen_once<'c, 'r>(
    context: &'c Context<'r>,
    a: &Dense<'c, 'r>,
    b: &Dense<'c, 'r>,
    link: Symmetry,
) -> Dense<'c, 'r> {
    if context.size() >= 7 && context.size() % 7 == 0 {
        strassen_subworld(context, a, b)
    } else {
        strassen_parent(context, a, b, link)
    }
}

fn run(context: &Context<'_>) {
    assert_eq!(N % 2, 0);
    let topology = Topology::new(vec![context.size()]);
    for link in [NS, AS, SY, SH] {
        let (a, b) = inputs(context, link);
        let reference = multiply(&a, &b, &topology);
        let result = strassen_once(context, &a, &b, link);
        let mut difference = reference.clone();
        difference
            .sum_from("ij", &result, "ij", topology.clone(), -1.0, 1.0)
            .unwrap();
        let norm = difference.norm2();
        let error = norm * norm / (N * N) as f64;
        // Preserve the source's strict err < 1.E-10 criterion.
        assert!(error.is_finite() && error < 1.0e-10);
    }
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);

    let rank = world.rank();
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_strassen: source one-level seven-product slicing, NS/AS/SY/SH, strict err<1e-10; parent and divisible-by-7 subworld branches; world+parity"
        );
    }
    world.close();
    drop(universe);
}
