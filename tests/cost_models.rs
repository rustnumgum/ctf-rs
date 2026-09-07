use ctf::cost::{Communication, Models};
#[test]
fn source_coefficients_round_trip() {
    let models = Models::upstream(4);
    assert_eq!(models.iter().count(), 32);
    assert_eq!(
        models.get("bcast_mdl").coefficients(),
        &[1.1115e-16, 1.0754e-16, 1.32e-9]
    );
    let mut bytes = Vec::new();
    models.write(&mut bytes).unwrap();
    let mut loaded = Models::upstream(4);
    loaded.get_mut("bcast_mdl").set_coefficients(&[0.; 3]);
    loaded.load(std::io::Cursor::new(bytes)).unwrap();
    for model in models.iter() {
        assert_eq!(
            loaded.get(model.name()).coefficients(),
            model.coefficients()
        );
    }
}
#[test]
fn communication_features_and_dispatch() {
    let mut models = Models::upstream(4);
    for (name, op, expected) in [
        ("bcast_mdl", Communication::Broadcast, 65.),
        ("red_mdl", Communication::Reduce { custom: false }, 125.),
        ("red_mdl_cst", Communication::Reduce { custom: true }, 125.),
        (
            "allred_mdl",
            Communication::AllReduce { custom: false },
            125.,
        ),
        (
            "allred_mdl_cst",
            Communication::AllReduce { custom: true },
            125.,
        ),
        ("alltoall_mdl", Communication::AllToAll, 485.),
        ("alltoallv_mdl", Communication::AllToAllV, 125.),
    ] {
        models.get_mut(name).set_coefficients(&[1., 2., 3.]);
        assert_eq!(models.communication(op, 4, 20), expected);
        assert_eq!(models.communication(op, 4, 0), 5.);
    }
    assert_eq!(models.communication(Communication::AllToAll, 1, 20), 1.);
}
#[test]
fn contraction_and_transpose_dispatch() {
    let mut models = Models::upstream(4);
    for (i, (name, custom, folded)) in [
        ("seq_tsr_ctr_mdl_cst", true, false),
        ("seq_tsr_ctr_mdl_cst_inr", true, true),
        ("seq_tsr_ctr_mdl_ref", false, false),
        ("seq_tsr_ctr_mdl_inr", false, true),
    ]
    .iter()
    .enumerate()
    {
        models.get_mut(name).set_coefficients(&[i as f64, 2., 3.]);
        assert_eq!(
            models.local_contraction(*custom, *folded, 10., 20.),
            i as f64 + 80.
        );
    }
    models
        .get_mut("non_contig_transp_mdl")
        .set_coefficients(&[1., 1.]);
    models
        .get_mut("shrt_contig_transp_mdl")
        .set_coefficients(&[2., 1.]);
    models
        .get_mut("long_contig_transp_mdl")
        .set_coefficients(&[3., 1.]);
    assert_eq!(models.transpose(&[3, 2], &[1, 0]), 7.);
    for leading in [4, 64, 65] {
        assert_eq!(
            models.transpose(&[leading, 2, 3], &[0, 2, 1]),
            (leading * 6) as f64 + if leading <= 64 { 2. } else { 3. }
        );
    }
    assert_eq!(models.transpose(&[], &[]), 0.);
    assert_eq!(models.transpose(&[3, 4], &[0, 1]), 0.);
}
