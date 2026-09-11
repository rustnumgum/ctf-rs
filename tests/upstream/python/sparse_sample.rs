// Adapted from pinned test/python/test_sparse.py::test_sample and
// src_python/ctf_ext.cxx::subsample. The source fixture is initially zero.
use ctf::{algebra::Arithmetic, context::Context, mapping::Distribution,
          random::Generator, tensor::Tensor};

fn run(context: &Context<'_>) {
    let a = Tensor::new(context, Distribution::cyclic(vec![4, 3, 5], context.size()),
                        Arithmetic::<f64>::new());
    let mut generator = Generator::new(context.rank() as u64);
    let norm = a.norm2();
    let mut a = a.into_sparse(|_| generator.unit_interval() < 0.5);
    let norm2 = a.norm2();
    a.sparsify(|_| generator.unit_interval() < 0.3);
    let norm3 = a.norm2();
    if context.rank() == 0 {
        println!("sparse_sample: norms={norm:e},{norm2:e},{norm3:e}; source nonincreasing bound");
    }
    assert!(norm2 <= norm);
    assert!(norm3 <= norm2);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    run(&world);
    let parity = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!("DIGIT / PASS sparse_sample: source zero (4,3,5), sample .5 then .3, nonincreasing norm2; world+parity");
    }
    world.close();
    drop(universe);
}
