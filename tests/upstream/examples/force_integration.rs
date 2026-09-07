//! Pinned examples/force_integration.cxx: dense antisymmetric force integration.

mod moldynamics;

use ctf::{
    algebra::{CustomMonoid, Monoid, Semiring},
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{AS, NS},
    tensor::Tensor,
};
use moldynamics::{Drand48, Force, ForceAlgebra, Particle, ParticleSet, acc_force};

type Particles<'c, 'r> = Tensor<'c, 'r, ParticleSet>;
type Forces<'c, 'r> = Tensor<'c, 'r, ForceAlgebra>;
type SymmetricForces<'c, 'r> = SymmetricTensor<'c, 'r, ForceAlgebra>;

fn symmetric_forces<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    links: Vec<ctf::symmetry::Symmetry>,
) -> SymmetricForces<'c, 'r> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !shape.is_empty() {
        mappings[0].augment_physical(&topology, 0);
    }
    if links
        .iter()
        .any(|&link| link != ctf::symmetry::Symmetry::NS)
    {
        for mapping in &mut mappings {
            mapping.augment_virtual(2 * context.size());
        }
    }
    SymmetricForces::new(
        context,
        SymmetricDistribution::new(Distribution::new(shape, topology, mappings), links),
        ForceAlgebra,
    )
}

fn summed_force<'c, 'r>(
    context: &'c Context<'r>,
    matrix: &SymmetricForces<'c, 'r>,
    n: usize,
) -> Forces<'c, 'r> {
    let algebra = ForceAlgebra;
    let mut sum = symmetric_forces(context, vec![n], vec![NS]);
    sum.sum_from("i", matrix, "ij", algebra.one(), algebra.zero());
    let rank = context.rank();
    let pairs: Vec<_> = sum
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| sum.distribution().distribution().owner(*key) == rank)
        .collect();
    let mut dense = Forces::new(
        context,
        Distribution::cyclic(vec![n], context.size()),
        algebra,
    );
    dense.write_add(&pairs);
    dense
}

fn apply(forces: &Forces<'_, '_>, particles: &mut Particles<'_, '_>) {
    particles.transform_from("i", forces, "i", forces, "i", |force, _, particle| {
        acc_force(*force, particle)
    });
}

fn reduce_pass(context: &Context<'_>, pass: &mut i32) {
    let minimum = CustomMonoid {
        identity: 1i32,
        addition: |left: &i32, right: &i32| (*left).min(*right),
    };
    context.all_reduce_monoid(&minimum, std::slice::from_mut(pass), true);
}

fn run(context: &Context<'_>) {
    let n = 5usize;
    let mut random = Drand48::new(context.rank());
    let mut particles = Particles::new(
        context,
        Distribution::cyclic(vec![n], context.size()),
        ParticleSet,
    );
    particles.transform(|_, particle| {
        *particle = Particle {
            dx: random.next(),
            dy: random.next(),
            coeff: 0.001 * random.next(),
            id: 777,
        };
    });
    let original = particles.local_pairs();

    let mut force = symmetric_forces(context, vec![n, n], vec![AS, NS]);
    force.transform(|_, value| {
        *value = Force {
            fx: random.next(),
            fy: random.next(),
        };
    });
    let mut force_twice = symmetric_forces(context, vec![n, n], vec![AS, NS]);
    let algebra = ForceAlgebra;
    force_twice.sum_from("ij", &force, "ij", algebra.one(), algebra.one());
    force_twice.sum_from("ij", &force, "ij", algebra.one(), algebra.one());

    let doubled_sum = summed_force(context, &force_twice, n);
    apply(&doubled_sum, &mut particles);
    let keys: Vec<_> = original.iter().map(|(key, _)| *key).collect();
    let changed = particles.read(&keys);
    let mut pass = 1i32;
    for ((_, old), new) in original.iter().zip(&changed) {
        if (old.dx - new.dx).abs() < 1e-6 && (old.dy - new.dy).abs() < 1e-6 {
            pass = 0;
        }
    }
    reduce_pass(context, &mut pass);

    force.transform(|_, value| *value = value.negated());
    let inverse_sum = summed_force(context, &force, n);
    apply(&inverse_sum, &mut particles);
    apply(&inverse_sum, &mut particles);
    let restored = particles.read(&keys);
    if pass == 1 {
        for ((_, reference), new) in original.iter().zip(&restored) {
            if (reference.dx - new.dx).abs() > 1e-6 || (reference.dy - new.dy).abs() > 1e-6 {
                pass = 0;
            }
        }
    }
    reduce_pass(context, &mut pass);
    assert_eq!(pass, 1);
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
            "DIGIT / PASS force_integration: dense AS force doubled then inverted twice; first modifies each particle, restored dx/dy within 1e-6; n=5; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
