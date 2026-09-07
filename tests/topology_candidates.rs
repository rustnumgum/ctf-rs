use ctf::{
    mapping::Topology,
    topology_candidates::{all_shapes, peel, peel_permutations},
};
fn dimensions(topologies: Vec<Topology>) -> Vec<Vec<usize>> {
    topologies.into_iter().map(|t| t.dimensions).collect()
}
#[test]
fn ordered_factorizations() {
    assert_eq!(dimensions(all_shapes(1)), vec![vec![]]);
    assert_eq!(dimensions(all_shapes(7)), vec![vec![7]]);
    assert_eq!(dimensions(all_shapes(4)), vec![vec![2, 2], vec![4]]);
    assert_eq!(
        dimensions(all_shapes(12)),
        vec![
            vec![2, 2, 3],
            vec![2, 3, 2],
            vec![2, 6],
            vec![4, 3],
            vec![3, 2, 2],
            vec![3, 4],
            vec![6, 2],
            vec![12]
        ]
    );
}
#[test]
fn ordered_folds() {
    assert_eq!(
        dimensions(peel(&Topology::new(vec![2, 3, 5]))),
        vec![vec![2, 3, 5], vec![6, 5], vec![2, 15], vec![30]]
    );
    assert_eq!(
        dimensions(peel_permutations(&Topology::new(vec![2, 3]))),
        vec![vec![2, 3], vec![6], vec![3, 2]]
    );
}
