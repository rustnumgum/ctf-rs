use ctf::{
    map_tensor::{Rejected, assign, coordinate_symmetry},
    mapping::{Mapping, Topology},
};

#[test]
fn longest_edge_and_restrictions() {
    let topo = Topology::new(vec![2, 3]);
    let mut maps = vec![Mapping::Unmapped; 2];
    let mut restricted = vec![false; 2];
    assign(
        &[12, 3],
        &topo,
        &[0, 1],
        &[false; 4],
        &mut restricted,
        &mut maps,
        true,
    )
    .unwrap();
    assert_eq!(maps[0].physical_phase(), 6);
    assert_eq!(maps[1].physical_phase(), 1);
    let mut maps = vec![Mapping::Unmapped; 2];
    assign(
        &[12, 3],
        &topo,
        &[0, 1],
        &[false; 4],
        &mut restricted,
        &mut maps,
        false,
    )
    .unwrap();
    assert_eq!(maps[0].physical_phase(), 2);
    assert_eq!(maps[1].physical_phase(), 3);
    assert_eq!(restricted, vec![true, true]);
}

#[test]
fn symmetry_virtualization() {
    let topo = Topology::new(vec![2, 3]);
    let mut maps = vec![Mapping::Unmapped; 2];
    assign(
        &[9, 9],
        &topo,
        &[0, 1],
        &[false, true, true, false],
        &mut [false; 2],
        &mut maps,
        true,
    )
    .unwrap();
    assert_eq!(
        maps.iter().map(Mapping::phase).collect::<Vec<_>>(),
        vec![6, 6]
    );
    assert_eq!(
        maps.iter().map(Mapping::physical_phase).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn rejected_candidates() {
    let topo = Topology::new(vec![2]);
    assert_eq!(
        assign(&[], &topo, &[0], &[], &mut [], &mut [], true),
        Err(Rejected::NoAssignableDimension)
    );
    let mut a = Mapping::Unmapped;
    a.augment_virtual(128);
    let mut b = Mapping::Unmapped;
    b.augment_virtual(129);
    assert_eq!(
        coordinate_symmetry(&mut [a, b], &[false, true, true, false]),
        Err(Rejected::PhaseLimit)
    );
}
