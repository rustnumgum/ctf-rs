use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    schedule::Schedule,
    tensor::Tensor,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

fn run(c: &Context<'_>) {
    let mut tensors = BTreeMap::new();
    for id in 0..6 {
        let mut t = Tensor::new(
            c,
            Distribution::cyclic(vec![3], c.size()),
            Arithmetic::<i64>::new(),
        );
        t.transform(|key, x| *x = if id == 0 { key as i64 + 1 } else { -100 });
        tensors.insert(id, t);
    }
    let observed = Rc::new(RefCell::new(Vec::new()));
    let mut schedule = Schedule::new(Some(2));
    for (output, scale) in [(1, 2), (2, 3)] {
        let observed = observed.clone();
        schedule.add_operation(&[0], output, 0.001, move |t| {
            let mut out = t.remove(&output).unwrap();
            observed.borrow_mut().push(out.context().size());
            out.sum_from(
                "i",
                &t[&0],
                "i",
                Topology::new(vec![out.context().size()]),
                scale,
                0,
            )
            .unwrap();
            t.insert(output, out);
        });
    }
    schedule.add_operation(&[1], 3, 0.001, |t| {
        let mut out = t.remove(&3).unwrap();
        out.sum_from(
            "i",
            &t[&1],
            "i",
            Topology::new(vec![out.context().size()]),
            1,
            0,
        )
        .unwrap();
        out.transform(|_, x| *x += 1);
        t.insert(3, out);
    });
    schedule.add_operation(&[], 1, 0.001, |t| {
        t.get_mut(&1).unwrap().transform(|_, x| *x = 7)
    });
    schedule.add_operation(&[], 1, 0.001, |t| {
        t.get_mut(&1).unwrap().transform(|_, x| *x = 9)
    });
    schedule.add_operation(&[1, 2], 4, 0.001, |t| {
        let mut out = t.remove(&4).unwrap();
        out.contract_from(
            "i",
            &t[&1],
            "i",
            &t[&2],
            "i",
            Topology::new(vec![out.context().size()]),
            1,
            0,
        )
        .unwrap();
        t.insert(4, out);
    });
    schedule.add_operation(&[4], 4, 0.001, |t| {
        t.get_mut(&4).unwrap().transform(|_, x| *x += 1)
    });
    schedule.add_operation(&[], 5, 0.001, |t| {
        t.get_mut(&5).unwrap().transform(|_, x| *x = 4)
    });
    for _ in 0..2 {
        let timing = schedule.execute(c, &mut tensors);
        for x in [
            timing.total_time,
            timing.exec_time,
            timing.comm_down_time,
            timing.comm_up_time,
            timing.imbalance_wall_time,
            timing.imbalance_accum_time,
        ] {
            assert!(x.is_finite() && x >= 0.0);
        }
        for (&id, t) in &tensors {
            assert_eq!(t.distribution(), &Distribution::cyclic(vec![3], c.size()));
            for (key, x) in t.local_pairs() {
                let base = key as i64 + 1;
                let expected = match id {
                    0 => base,
                    1 => 9,
                    2 => 3 * base,
                    3 => 2 * base + 1,
                    4 => 27 * base + 1,
                    5 => 4,
                    _ => unreachable!(),
                };
                assert_eq!(x, expected, "tensor {id}, key {key}");
            }
        }
    }
    if c.size() > 1 {
        assert!(
            observed.borrow().iter().any(|&size| size < c.size()),
            "independent operations must execute on proper subcommunicators"
        );
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
            "DIGIT / PASS schedule: exact distributed sum/contract, RAW/WAR/WAW/self updates, no-input roots, subsecond partition costs, replay, world+parity"
        );
    }
    world.close();
    drop(universe);
}
