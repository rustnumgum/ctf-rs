use ctf::node_aware::inter_node_grids;

#[test]
fn scalar_topology() {
    assert_eq!(inter_node_grids(&[],1),vec![Vec::<usize>::new()]);
}

#[test]
fn exact_branch_order() {
    assert_eq!(inter_node_grids(&[4,4],4),vec![vec![4,1],vec![2,2],vec![1,4]]);
    assert_eq!(inter_node_grids(&[6,4],6),vec![vec![6,1],vec![3,2]]);
    assert_eq!(inter_node_grids(&[2,3],6),vec![vec![2,3]]);
    assert_eq!(inter_node_grids(&[4,4],1),vec![vec![1,1]]);
    assert_eq!(inter_node_grids(&[1],1),vec![vec![1]]);
}

#[test]
fn grid_invariants() {
    for (grid,nodes) in [(vec![8,4,3],12),(vec![2,2,2],4),(vec![5,2],5)] {
        let candidates = inter_node_grids(&grid,nodes);
        assert!(!candidates.is_empty());
        for (i,candidate) in candidates.iter().enumerate() {
            assert_eq!(candidate.iter().product::<usize>(),nodes);
            assert!(candidate.iter().zip(&grid).all(|(&n,&p)|p%n == 0));
            assert!(!candidates[..i].contains(candidate));
        }
    }
}
