use ctf::{cost::Models,folded_cost,mapping::{Distribution,Mapping,Topology},normal_mapping::Problem,plan_cost::Tree};
fn models()->Models{let mut m=Models::upstream(1);
    for name in ["seq_tsr_ctr_mdl_inr","bcast_mdl","red_mdl","dgtog_res_mdl"]{m.get_mut(name).set_coefficients(&[0.,0.,1.]);}
    for name in ["blres_mdl","non_contig_transp_mdl","shrt_contig_transp_mdl","long_contig_transp_mdl"]{m.get_mut(name).set_coefficients(&[0.,1.]);}
    m
}
fn local(shape:&[usize])->Distribution{Distribution::new(shape.to_vec(),Topology::new(vec![]),vec![Mapping::Unmapped;shape.len()])}
#[test]
fn folded_two_dimensional_tree_adds_source_resident_buffers(){
    let shapes:[&[usize];3]=[&[3,4],&[4,5],&[3,5]];let indices=["ik","kj","ij"];
    let mapped=Problem::new(shapes,indices).unwrap().map_to_topology(&Topology::new(vec![2,2]),0,[None;3]).unwrap();
    let old=shapes.map(|s|Distribution::cyclic(s.to_vec(),4));
    let e=folded_cost::estimate_dense_folded(old.each_ref(),mapped.each_ref(),indices,&models(),8,&[1,2],false).unwrap().unwrap();
    let Tree::Panels{steps,panel_bytes,child,..}=&e.tree else{panic!("lost 2D communication")};
    assert_eq!(*steps,2);assert_eq!(*panel_bytes,[32,48,0]);
    let Tree::Local{folded,operand_bytes,flops,..}=child.as_ref()else{panic!("missing folded leaf")};
    assert!(*folded);assert_eq!(*operand_bytes,[32,48,48]);assert_eq!(*flops,24.);
    assert_eq!(e.inner.seconds,208.);assert_eq!(e.inner.working_bytes,128);
    assert_eq!(e.fold_resident_bytes,128);assert_eq!(e.fold_temporary_bytes,0);
    assert_eq!(e.redistribution_seconds,[64.,96.,192.]);assert_eq!(e.redistributed_input_bytes,80);
    assert_eq!(e.redistribution_temporary_bytes,192);assert_eq!(e.memory_bytes,336);assert_eq!(e.seconds,560.);
}
#[test]
fn partial_fold_residual_cost_and_source_batch_model_omission(){
    let mapped=[local(&[2,3,4]),local(&[4,5]),local(&[3,5])];
    let e=folded_cost::estimate_dense_folded(mapped.each_ref(),mapped.each_ref(),["xik","kj","ij"],&models(),8,&[],false).unwrap().unwrap();
    assert_eq!(e.transpose_seconds,[24.,0.,0.]);assert_eq!(e.inner.seconds,240.);
    assert_eq!(e.fold_resident_bytes,472);assert_eq!(e.memory_bytes,472);assert_eq!(e.seconds,264.);
    let mapped=[local(&[3,4,2]),local(&[4,5,2]),local(&[3,5,2])];
    let e=folded_cost::estimate_dense_folded(mapped.each_ref(),mapped.each_ref(),["ikl","kjl","ijl"],&models(),8,&[],false).unwrap().unwrap();
    assert_eq!(e.descriptor.batches,2);
    // The pinned est_membw/est_fp multiply by mnk but omit l. Preserve this
    // estimator quirk; actual BLAS execution still processes both batches.
    let Tree::Local{operand_bytes,flops,..}=e.tree else{panic!("unexpected communication")};
    assert_eq!(operand_bytes,[96,160,120]);assert_eq!(flops,120.);
    assert_eq!(e.inner.seconds,120.);assert_eq!(e.fold_resident_bytes,752);assert_eq!(e.memory_bytes,752);
    let scalar=[local(&[]),local(&[]),local(&[])];
    assert!(folded_cost::estimate_dense_folded(scalar.each_ref(),scalar.each_ref(),["","",""],&models(),8,&[],false).unwrap().is_none());
}
