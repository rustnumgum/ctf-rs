//! Exact dense semantic acceptance, including the upstream diag_ctr identity.
use ctf::{
    algebra::Arithmetic,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let np = world.size();
    let topo = Topology::new(if np == 4 { vec![2, 2] } else { vec![np] });
    let make = |shape: Vec<usize>| {
        Tensor::new(
            &world,
            Distribution::cyclic(shape, np),
            Arithmetic::<i64>::new(),
        )
    };
    let mut a = make(vec![3, 5, 3, 5]);
    a.transform(|key, v| *v = (key + 1) as i64);
    let (diag, labels) = a.extract_diagonal("aiai");
    assert_eq!(labels, "ai");
    for (key, value) in diag.local_pairs() {
        assert_eq!(value, 1 + 16 * (key % 3) as i64 + 48 * (key / 3) as i64);
    }
    let mut ma = make(vec![3, 5]);
    ma.sum_from("ai", &a, "aiai", topo.clone(), 1, 0).unwrap();
    let mut trace = make(vec![]);
    trace.sum_from("", &a, "aiai", topo.clone(), 1, 0).unwrap();
    assert_eq!(trace.reduce(), 1695);
    assert_eq!(ma.reduce(), 1695);
    let mut matrix = make(vec![3, 3]);
    matrix.transform(|key, v| *v = (key + 2) as i64);
    let mut vector = make(vec![3]);
    vector.transform(|key, v| *v = (key + 1) as i64);
    let mut output = make(vec![3, 3]);
    output.transform(|_, v| *v = 7);
    output
        .contract_from("ii", &matrix, "ii", &vector, "i", topo.clone(), 2, 3)
        .unwrap();
    for (key, value) in output.local_pairs() {
        assert_eq!(
            value,
            if key % 3 == key / 3 {
                2 * ((key + 2) * (key % 3 + 1)) as i64 + 21
            } else {
                7
            }
        );
    }
    output
        .sum_from("ii", &vector, "i", topo.clone(), 1, 0)
        .unwrap();
    for (key, value) in output.local_pairs() {
        assert_eq!(
            value,
            if key % 3 == key / 3 {
                (key % 3 + 1) as i64
            } else {
                7
            }
        );
    }
    let mut squared = make(vec![]);
    squared
        .sum_function_from("", &vector, "i", topo.clone(), 2, 0, |v| v * v)
        .unwrap();
    assert_eq!(squared.reduce(), 56);
    let empty = make(vec![0, 0]);
    trace.transform(|_, v| *v = 2);
    trace.sum_from("", &empty, "ii", topo, 1, 3).unwrap();
    assert_eq!(trace.reduce(), 6);
    let mut source = make(vec![7, 4]);
    source.transform(|key, v| *v = (key + 1) as i64);
    let mut dest = make(vec![8, 6]);
    dest.transform(|_, v| *v = 9);
    dest.assign_slice(&[3..7, 1..4], &source, &[1..5, 0..3], &2, &3);
    for (key, value) in dest.local_pairs() {
        let (x, y) = (key % 8, key / 8);
        assert_eq!(
            value,
            if (3..7).contains(&x) && (1..4).contains(&y) {
                27 + 2 * (x - 1 + 7 * (y - 1)) as i64
            } else {
                9
            }
        );
    }
    dest.assign_slice(&[3..3, 1..4], &source, &[1..1, 0..3], &2, &0);
    let mut written = make(vec![2]);
    written.transform(|_, v| *v = 5);
    written.write_scaled(&[(0, 1), (0, 2)], &2, &3);
    assert_eq!(written.read(&[0, 1]), vec![15 + 6 * np as i64, 5]);
    drop(written);
    drop(dest);
    drop(source);
    drop(empty);
    drop(squared);
    drop(output);
    drop(vector);
    drop(matrix);
    drop(trace);
    drop(ma);
    drop(diag);
    drop(a);
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS dense_semantics: diagonal extraction/trace/repeated output, indexed contraction, nonlinear sum, slice insertion, affine writes; ranks={np}"
        );
    }
    world.close();
    drop(universe);
}
