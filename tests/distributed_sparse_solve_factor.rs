use ctf::{algebra::Arithmetic, context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology}, sparse::SparseTensor, tensor::Tensor};

fn coordinates(key: usize) -> [usize; 3] { [key % 3, key / 3 % 3, key / 9] }
fn weight(key: usize) -> f64 {
    let c = coordinates(key);
    if c[0] == c[1] { 1. + c[2] as f64 } else { 0. }
}
fn solution(row: usize, auxiliary: usize) -> f64 { 0.5 + row as f64 + auxiliary as f64 * 0.25 }
fn gram(mode: usize, row: usize) -> [[f64; 2]; 2] {
    let mut g = [[0.; 2]; 2];
    for key in 0..18 {
        let c = coordinates(key);
        if c[mode] != row { continue; }
        let v = [1., (0..3).filter(|&m| m != mode).map(|m| (c[m] + 1) as f64).product()];
        for i in 0..2 { for j in 0..2 { g[i][j] += weight(key) * v[i] * v[j]; } }
    }
    g
}
fn run(c: &Context<'_>) {
    for mapped_mode in 0..4 {
        let topology = Topology::new(vec![c.size()]);
        let mut mapping = Mapping::Unmapped;
        mapping.augment_physical(&topology, 0);
        mapping.augment_virtual(2 * c.size());
        let mut mappings = vec![Mapping::Unmapped; 3];
        if mapped_mode < 3 { mappings[mapped_mode] = mapping; }
        let d = Distribution::new(vec![3, 3, 2], topology, mappings);
        let entries: Vec<_> = (0..18).filter(|&key| (weight(key) != 0. || key == 1) && d.owner(key) == c.rank())
            .map(|key| (key, weight(key))).collect();
        let mut weights = SparseTensor::new(c, d.clone(), Arithmetic::<f64>::new());
        weights.write_add(&entries);
        for mode in 0..3 {
            let factors: Vec<_> = (0..3).filter(|&m| m != mode).map(|m| {
                let mut t = Tensor::new(c, Distribution::cyclic(vec![2, d.shape[m]], c.size()), Arithmetic::<f64>::new());
                t.transform(|key, v| *v = if key % 2 == 0 { 1. } else { (key / 2 + 1) as f64 });
                t
            }).collect();
            let mut rhs = Tensor::new(c, Distribution::cyclic(vec![2, d.shape[mode]], c.size()), Arithmetic::<f64>::new());
            rhs.transform(|key, v| { let row = key / 2; let g = gram(mode, row);
                *v = (0..2).map(|j| g[key % 2][j] * solution(row, j)).sum(); });
            let result = weights.solve_factor(mode, &factors.iter().collect::<Vec<_>>(), &rhs).unwrap();
            assert_eq!(result.distribution(), rhs.distribution());
            for (key, actual) in result.local_pairs() {
                let expected = solution(key / 2, key % 2);
                assert!(actual.is_finite() && (actual - expected).abs() <= 1e-8 + 1e-5 * expected.abs());
            }
            // Missing sparse rows create singular normal systems; surface POSV INFO.
            let empty = SparseTensor::new(c, d.clone(), Arithmetic::<f64>::new());
            match empty.solve_factor(mode, &factors.iter().collect::<Vec<_>>(), &rhs) {
                Err(info) => assert_eq!(info, 1),
                Ok(_) => panic!("empty sparse normal system accepted"),
            }
        }
    }
}
fn main() {
    let runtime = Runtime::initialize(); let world = runtime.world(); run(&world);
    let child = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&child); child.close();
    if world.rank() == 0 { println!("DIGIT / PASS distributed_sparse_solve_factor: weighted stored-entry Gram systems, all output and physical modes, virtual blocks, empty shards/systems, world+parity; atol=1e-8 rtol=1e-5"); }
    world.close(); runtime.finalize();
}
