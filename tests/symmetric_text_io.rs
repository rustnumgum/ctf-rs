use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{AS, NS, SH, SY},
    tensor::Tensor,
};
macro_rules! scalar_case {
    ($name:ident,$t:ty) => {
        fn $name(c: &Context<'_>, token: u64, scope: usize) {
            for (kind_id, kind) in [SY, AS, SH].into_iter().enumerate() {
                for reverse in [false, true] {
                    let top = Topology::new(vec![c.size()]);
                    let mut first = Mapping::Unmapped;
                    first.augment_physical(&top, 0);
                    first.augment_virtual(2 * c.size());
                    let mut second = Mapping::Unmapped;
                    second.augment_virtual(2 * c.size());
                    let distribution = SymmetricDistribution::new(
                        Distribution::new(vec![3, 3], top, vec![first, second]),
                        vec![kind, NS],
                    );
                    let path = std::env::temp_dir().join(format!(
                        "ctf-sym-io-{token}-{scope}-{}-{kind_id}-{reverse}.txt",
                        stringify!($name)
                    ));
                    let mut original =
                        SymmetricTensor::new(c, distribution.clone(), Arithmetic::<$t>::new());
                    original.transform(|key, x| *x = (key % 3 + 1) as $t);
                    original.write_sparse_to_file(&path, true, reverse);
                    let mut restored =
                        SymmetricTensor::new(c, distribution.clone(), Arithmetic::<$t>::new());
                    restored.read_sparse_from_file(&path, true, reverse);
                    assert_eq!(restored.local_pairs(), original.local_pairs());
                    // The pinned writer exports canonical nonzero pairs, not the full orbit.
                    let mut packed = Tensor::new(
                        c,
                        Distribution::cyclic(vec![3, 3], c.size()),
                        Arithmetic::<$t>::new(),
                    );
                    packed.read_sparse_from_file(&path, true, reverse);
                    for (key, x) in packed.local_pairs() {
                        let expected = if distribution.canonicalize(key) == Some((key, 1)) {
                            (key % 3 + 1) as $t
                        } else {
                            0 as $t
                        };
                        assert_eq!(x, expected);
                    }
                    c.barrier();
                    if c.rank() == 0 {
                        let records = if reverse {
                            "1 0 2\n0 1 1\n"
                        } else {
                            "0 1 2\n1 0 1\n"
                        };
                        std::fs::write(&path, records).unwrap();
                    }
                    c.barrier();
                    let mut summed =
                        SymmetricTensor::new(c, distribution.clone(), Arithmetic::<$t>::new());
                    summed.read_sparse_from_file(&path, true, reverse);
                    for (key, x) in summed.local_pairs() {
                        assert_eq!(
                            x,
                            if key == 3 {
                                if kind == AS { 1 as $t } else { 3 as $t }
                            } else {
                                0 as $t
                            }
                        );
                    }
                    c.barrier();
                    if c.rank() == 0 {
                        std::fs::remove_file(&path).unwrap();
                    }
                    c.barrier();
                }
            }
        }
    };
}
scalar_case!(real32, f32);
scalar_case!(real64, f64);
scalar_case!(int32, i32);
scalar_case!(int64, i64);
fn run(c: &Context<'_>, token: u64, scope: usize) {
    real32(c, token, scope);
    real64(c, token, scope);
    int32(c, token, scope);
    int64(c, token, scope);
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let mut token = [std::process::id() as u64];
    world.broadcast(0, &mut token);
    run(&world, token[0], 0);
    let color = world.rank() % 2;
    let child = world
        .split(Some(color as i32), world.rank() as i32)
        .unwrap();
    run(&child, token[0], color + 1);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS symmetric_text_io: four types, canonical SY/AS/SH export, reverse indices, signed orbit accumulation; exact world+parity"
        );
    }
    world.close();
    drop(universe);
}
