//! Pinned `examples/force_integration_sparse.cxx` with compressed AS storage
//! and selected direct custom accumulation.

mod moldynamics;

use std::mem::size_of;

use ctf::{
    algebra::{Arithmetic, CustomMonoid, Semiring},
    context::Context,
    cost::Models,
    mapping::{Distribution, Mapping, Topology},
    sparse_symmetric::SparseSymmetricTensor,
    sparse_symmetric_search::{self, Options},
    symmetric_distribution::SymmetricDistribution,
    symmetry::Symmetry::{AS, NS},
    tensor::Tensor,
    topology_candidates,
};
use moldynamics::{
    Drand48, Force, ForceAlgebra, Particle, ParticleSet, acc_force, get_distance, get_force,
};

const N: usize = 5;

type Particles<'c, 'r> = Tensor<'c, 'r, ParticleSet>;
type Forces<'c, 'r> = SparseSymmetricTensor<'c, 'r, ForceAlgebra>;

fn force_distribution(context: &Context<'_>) -> SymmetricDistribution {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; 2];
    mappings[0].augment_physical(&topology, 0);
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricDistribution::new(
        Distribution::new(vec![N, N], topology, mappings),
        vec![AS, NS],
    )
}

fn nonzeros(force: &Forces<'_, '_>) -> u64 {
    force
        .context()
        .all_reduce(&Arithmetic::<u64>::new(), &(force.local_nnz() as u64))
}

fn reduce(context: &Context<'_>, value: i32, maximum: bool) -> i32 {
    let algebra = CustomMonoid {
        identity: if maximum { 0 } else { 1 },
        addition: move |left: &i32, right: &i32| {
            if maximum {
                (*left).max(*right)
            } else {
                (*left).min(*right)
            }
        },
    };
    context.all_reduce(&algebra, &value)
}

fn run(context: &Context<'_>) {
    let mut random = Drand48::new(context.rank());
    let mut particles = Particles::new(
        context,
        Distribution::cyclic(vec![N], context.size()),
        ParticleSet,
    );
    particles.transform(|key, particle| {
        let (dx, dy) = if key == 0 {
            (0.5, 0.5)
        } else {
            (random.next(), random.next())
        };
        *particle = Particle {
            dx,
            dy,
            coeff: 0.001 * random.next(),
            id: 777,
        };
    });

    let rank = context.rank();
    let original: Vec<_> = particles
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| particles.distribution().owner(*key) == rank)
        .collect();
    // The source explicitly gathers P only to construct its random fixture.
    let all = particles.read(&(0..N).collect::<Vec<_>>());
    let mut pairs = Vec::new();
    for (i, particle) in &original {
        for (j, other) in all.iter().enumerate() {
            if j != *i && get_distance(*particle, *other) < 0.708 {
                // Preserve the source's intentionally transposed raw key.
                pairs.push((*i * N + j, get_force(*particle, *other)));
            }
        }
    }

    let distribution = force_distribution(context);
    let mut force = Forces::new(context, distribution.clone(), ForceAlgebra);
    force.write_add(&pairs);
    let mut force_twice = Forces::new(context, distribution, ForceAlgebra);
    let algebra = ForceAlgebra;
    force_twice.sum_from("ij", &force, "ij", algebra.one(), algebra.one());
    force_twice.sum_from("ij", &force, "ij", algebra.one(), algebra.one());

    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let selected = sparse_symmetric_search::search(
        context,
        force_twice.distribution(),
        particles.distribution(),
        ["ij", "i"],
        &catalog,
        &models,
        nonzeros(&force_twice),
        size_of::<Force>(),
        size_of::<u64>() + size_of::<Force>(),
        size_of::<Particle>(),
        Options {
            memory_limit: u64::MAX,
        },
    )
    .unwrap()
    .expect("automatic compressed sparse transform search found no valid mapping");

    particles.accumulate_sparse_symmetric_from_selected(
        "i",
        &force_twice,
        "ij",
        &selected,
        |value, particle| acc_force(*value, particle),
    );
    let keys: Vec<_> = original.iter().map(|(key, _)| *key).collect();
    let changed = particles.read(&keys);
    let local_changed = original
        .iter()
        .zip(&changed)
        .any(|((_, old), new)| {
            (old.dx - new.dx).abs() > 1e-6 || (old.dy - new.dy).abs() > 1e-6
        }) as i32;
    let mut pass = reduce(context, local_changed, true);

    force.add_inverse();
    particles.accumulate_sparse_symmetric_from_selected(
        "i",
        &force,
        "ij",
        &selected,
        |value, particle| acc_force(*value, particle),
    );
    particles.accumulate_sparse_symmetric_from_selected(
        "i",
        &force,
        "ij",
        &selected,
        |value, particle| acc_force(*value, particle),
    );
    let restored = particles.read(&keys);
    if pass == 1
        && original.iter().zip(&restored).any(|((_, old), new)| {
            (old.dx - new.dx).abs() > 1e-6 || (old.dy - new.dy).abs() > 1e-6
        })
    {
        pass = 0;
    }
    pass = reduce(context, pass, false);
    assert_eq!(pass, 1);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS force_integration_sparse: one selected compressed-AS uacc changes any particle by >1e-6; twice inverse uacc restores every dx/dy within 1e-6; n=5; world+parity"
        );
    }
    world.close();
    drop(universe);
}
