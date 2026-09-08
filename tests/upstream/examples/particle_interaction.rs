//! Pinned examples/particle_interaction.cxx: dense pair forces and integration.

mod moldynamics;

use ctf::{
    algebra::{Arithmetic, CustomMonoid},
    context::Context,
    mapping::Distribution,
    tensor::Tensor,
};
use moldynamics::{Drand48, Force, ForceAlgebra, Particle, ParticleSet, acc_force, get_force};

type Particles<'c, 'r> = Tensor<'c, 'r, ParticleSet>;
type Forces<'c, 'r> = Tensor<'c, 'r, ForceAlgebra>;

fn replicated_particles(particles: &Particles<'_, '_>, n: usize) -> Vec<Particle> {
    let context = particles.context();
    let distribution = particles.distribution();
    let mut values = vec![Particle::default(); n];
    for (key, particle) in particles.local_pairs() {
        if distribution.owner(key) == context.rank() {
            values[key] = particle;
        }
    }
    for (key, particle) in values.iter_mut().enumerate() {
        context.broadcast(distribution.owner(key), std::slice::from_mut(particle));
    }
    values
}

fn apply(forces: &Forces<'_, '_>, particles: &mut Particles<'_, '_>) {
    particles.transform_from("i", forces, "i", forces, "i", |force, _, particle| {
        acc_force(*force, particle)
    });
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

    // This is the source contraction's missing-axis replication: each owner
    // broadcasts its particle. It does not gather the tensor to a root.
    let all_particles = replicated_particles(&particles, n);
    let mut force = Forces::new(
        context,
        Distribution::cyclic(vec![n], context.size()),
        ForceAlgebra,
    );
    force.transform(|i, value| {
        let mut total = Force::default();
        for &other in &all_particles {
            let pair = get_force(all_particles[i], other);
            total.fx += pair.fx;
            total.fy += pair.fy;
        }
        *value = total;
    });

    let mut force_all = Forces::new(
        context,
        Distribution::cyclic(vec![n, n], context.size()),
        ForceAlgebra,
    );
    force_all.transform(|key, value| {
        let i = key % n;
        let j = key / n;
        *value = get_force(all_particles[i], all_particles[j]);
    });

    let mut magnitude = Tensor::new(
        context,
        Distribution::cyclic(vec![n], context.size()),
        Arithmetic::<f64>::new(),
    );
    let rank = context.rank();
    let all_contributions: Vec<_> = force_all
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| force_all.distribution().owner(*key) == rank)
        .map(|(key, value)| (key % n, value.fx + value.fy))
        .collect();
    magnitude.write_add(&all_contributions);
    let reduced_contributions: Vec<_> = force
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| force.distribution().owner(*key) == rank)
        .map(|(key, value)| (key, -(value.fx + value.fy)))
        .collect();
    magnitude.write_add(&reduced_contributions);

    let mut pass = i32::from(magnitude.norm2() < 1e-6);
    let minimum = CustomMonoid {
        identity: 1i32,
        addition: |left: &i32, right: &i32| (*left).min(*right),
    };
    context.all_reduce_monoid(&minimum, std::slice::from_mut(&mut pass), true);
    assert_eq!(pass, 1);

    apply(&force, &mut particles);
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
            "DIGIT / PASS particle_interaction: reduced pair force equals NS force-matrix row reduction; n=5; magnitude residual norm2<1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
