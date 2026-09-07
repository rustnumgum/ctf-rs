use ctf::{
    cost::Models,
    mapping::{Distribution, Mapping, Topology},
    normal_mapping::Problem,
    sparse_cost::{Fractions, Storage},
    sparse_mapped_cost::{self, Inputs, Tree, Unsupported},
};

fn models() -> Models {
    let mut models = Models::upstream(1);
    for (name, coefficients) in [
        ("seq_tsr_spctr_k0", &[1., 2., 3.][..]),
        ("bcast_mdl", &[1., 2., 3.]),
        ("pin_keys_mdl", &[5., 7.]),
        ("spredist_mdl", &[1., 2., 3.]),
        ("blres_mdl", &[1., 2.]),
    ] {
        models.get_mut(name).set_coefficients(coefficients);
    }
    models
}

fn inputs(a_dense_virtual_size: usize) -> Inputs {
    Inputs {
        storage: [
            Storage { sparse: true, element_size: 8, pair_size: 16,
                dense_virtual_size: a_dense_virtual_size, custom_addition: false },
            Storage { sparse: false, element_size: 8, pair_size: 16,
                dense_virtual_size: 0, custom_addition: false },
            Storage { sparse: false, element_size: 8, pair_size: 16,
                dense_virtual_size: 0, custom_addition: false },
        ],
        fractions: Fractions { a: 0.5, b: 1., c: 1. },
        custom: false,
    }
}

#[test]
fn aligned_replication_virtual_and_redistribution() {
    let topology = Topology::new(vec![2]);
    let physical = || Mapping::Physical { axis: 0, processes: 2,
        child: Box::new(Mapping::Unmapped) };
    let virtual_two = || Mapping::Virtual { copies: 2,
        child: Box::new(Mapping::Unmapped) };
    let mapped = [
        Distribution::new(vec![4, 4], topology.clone(), vec![physical(), virtual_two()]),
        Distribution::new(vec![4, 4], topology.clone(), vec![virtual_two(), Mapping::Unmapped]),
        Distribution::new(vec![4, 4], topology, vec![physical(), Mapping::Unmapped]),
    ];
    let plan = sparse_mapped_cost::build_unfolded(
        mapped.each_ref(), ["ik", "kj", "ij"], inputs(4),
    ).unwrap();
    let Tree::Pin(pin, child) = &plan.tree else { panic!("missing source key pin") };
    assert_eq!(pin.dense_block_size, 8);
    let Tree::Replicate(replication, child) = child.as_ref() else {
        panic!("missing source replication")
    };
    assert!(replication.a.communicator_ranks.is_empty());
    assert_eq!(replication.b.communicator_ranks, vec![2]);
    assert!(replication.c.communicator_ranks.is_empty());
    assert_eq!(replication.b.size, 16);
    let Tree::Virtual(virtual_level, child) = child.as_ref() else {
        panic!("missing source virtual layer")
    };
    assert_eq!(virtual_level.dimensions, vec![1, 2, 1]);
    let Tree::Local(local) = child.as_ref() else { panic!("missing source k0 leaf") };
    assert!(matches!(&local.kernel,
        ctf::sparse_cost::local::Kernel::General { extents } if extents == &vec![2, 2, 4]));

    let estimate = plan.estimate(&models(), 1);
    assert_eq!(estimate.seconds, 1862.);
    assert_eq!(estimate.working_bytes, 392);

    let old = [
        Distribution::cyclic(vec![4, 4], 2),
        Distribution::cyclic(vec![4, 4], 2),
        Distribution::cyclic(vec![4, 4], 2),
    ];
    let total = plan.estimate_with_redistribution(
        old.each_ref(), mapped.each_ref(), &models(), 1,
    );
    assert_eq!(total.redistribution[0].seconds, 99.);
    assert_eq!(total.redistribution[1].seconds, 259.);
    assert_eq!(total.redistribution[2].seconds, 0.);
    assert_eq!(total.seconds, 2220.);
    assert_eq!(total.memory_bytes, 584);
}

#[test]
fn mismatched_gemm_mapping_builds_sparse_panel() {
    let shapes: [&[usize]; 3] = [&[3, 4], &[4, 5], &[3, 5]];
    let indices = ["ik", "kj", "ij"];
    let mapped = Problem::new(shapes, indices).unwrap()
        .map_to_topology(&Topology::new(vec![2, 2]), 0, [None; 3]).unwrap();
    let plan = sparse_mapped_cost::build_unfolded(
        mapped.each_ref(), indices, inputs(4),
    ).unwrap();
    let Tree::Pin(_, child) = &plan.tree else { panic!("missing source key pin") };
    let Tree::Panel(panel, child) = child.as_ref() else { panic!("missing source k panel") };
    assert_eq!(panel.edge, 2);
    assert_eq!((panel.a.moving, panel.a.ranks, panel.a.outer, panel.a.inner),
        (true, 2, 1, 1));
    assert_eq!((panel.b.moving, panel.b.ranks, panel.b.outer, panel.b.inner),
        (true, 2, 1, 6));
    assert_eq!((panel.c.moving, panel.c.outer, panel.c.inner), (false, 1, 0));
    assert!(matches!(child.as_ref(), Tree::Local(_)));
    let estimate = plan.estimate(&models(), 1);
    assert_eq!(estimate.seconds, 1753.);
    assert_eq!(estimate.working_bytes, 240);
}

#[test]
fn unfolded_scope_rejects_unsupported_storage_and_labels() {
    let topology = Topology::new(vec![1]);
    let scalar = || Distribution::new(vec![], topology.clone(), vec![]);
    let mapped = [scalar(), scalar(), scalar()];
    let mut unsupported = inputs(1);
    unsupported.storage[1].sparse = true;
    assert_eq!(sparse_mapped_cost::build_unfolded(
        mapped.each_ref(), ["", "", ""], unsupported,
    ).unwrap_err(), Unsupported::Storage);

    let a = Distribution::new(vec![2], topology.clone(), vec![Mapping::Unmapped]);
    assert_eq!(sparse_mapped_cost::build_unfolded(
        [&a, &mapped[1], &mapped[2]], ["x", "", ""], inputs(2),
    ).unwrap_err(), Unsupported::AOnlyLabel);
}
