use ctf::{cost::Models,mapping::{Distribution,Mapping,Topology},mapped_cost,normal_mapping::Problem,plan_cost::Tree,mapping_variants::VariantSpace};
fn models()->Models{
    let mut m=Models::upstream(1);
    for name in ["seq_tsr_ctr_mdl_ref","bcast_mdl","red_mdl","dgtog_res_mdl"]{
        m.get_mut(name).set_coefficients(&[0.,0.,1.]);
    }
    m.get_mut("red_mdl_cst").set_coefficients(&[0.,0.,3.]);m
}
#[test]
fn paired_k_panels_and_redistribution_total(){
    let shapes:[&[usize];3]=[&[3,4],&[4,5],&[3,5]];let indices=["ik","kj","ij"];
    let mapped=Problem::new(shapes,indices).unwrap().map_to_topology(&Topology::new(vec![2,2]),0,[None;3]).unwrap();
    let tree=mapped_cost::dense_unfolded(mapped.each_ref(),indices,8,&[1.,2.],false,false);
    let Tree::Panels{steps,panel_bytes,movement,child,..}=&tree else{panic!("source k panel layer missing")};
    assert_eq!(*steps,2);assert_eq!(*panel_bytes,[32,48,0]);
    assert_eq!(movement[0].as_ref().unwrap().nodes,1.);assert_eq!(movement[1].as_ref().unwrap().nodes,2.);assert!(movement[2].is_none());
    let Tree::Local{operand_bytes,flops,..}=child.as_ref()else{panic!("unexpected virtual layer")};
    assert_eq!(*operand_bytes,[32,48,48]);assert_eq!(*flops,24.);
    let m=models();let estimate=tree.estimate(&m,1);
    assert_eq!(estimate.seconds,208.);assert_eq!(estimate.working_bytes,128);assert_eq!(estimate.internode_volume,256.);
    let old=shapes.map(|shape|Distribution::cyclic(shape.to_vec(),4));
    let total=mapped_cost::estimate_dense_unfolded(old.each_ref(),mapped.each_ref(),indices,&m,8,&[1.,2.],false,false);
    assert_eq!(total.redistribution_seconds,[64.,96.,192.]);assert_eq!(total.redistributed_input_bytes,80);
    assert_eq!(total.redistribution_temporary_bytes,192);assert_eq!(total.memory_bytes,272);assert_eq!(total.seconds,560.);
}
#[test]
fn output_movement_uses_custom_reduction_and_rectangular_k_uses_lcm(){
    let topology=Topology::new(vec![2,2]);
    let p=|axis|Mapping::Physical{axis,processes:2,child:Box::new(Mapping::Unmapped)};
    let mapped=[Distribution::new(vec![3,4],topology.clone(),vec![p(0),p(1)]),
        Distribution::new(vec![4,5],topology.clone(),vec![p(1),p(0)]),
        Distribution::new(vec![3,5],topology,vec![p(1),p(0)])];
    let tree=mapped_cost::dense_unfolded(mapped.each_ref(),["ik","kj","ij"],8,&[1.,2.],false,true);
    let Tree::Panels{panel_bytes,movement,..}=&tree else{panic!("source i panel layer missing")};
    assert_eq!(*panel_bytes,[32,0,48]);assert!(movement[1].is_none());assert!(movement[2].is_some());
    assert_eq!(tree.estimate(&models(),1).seconds,400.);
    let variant=VariantSpace::new([&[3,4],&[4,5],&[3,5]],["ik","kj","ij"],Topology::new(vec![2,3])).unwrap().decode(3).unwrap();
    let tree=mapped_cost::dense_unfolded(variant.distributions.each_ref(),["ik","kj","ij"],8,&[1.,1.],false,false);
    let Tree::Panels{steps,panel_bytes,..}=&tree else{panic!("rectangular panel layer missing")};
    assert_eq!(*steps,6);assert_eq!(*panel_bytes,[8,24,0]);
    let estimate=tree.estimate(&models(),1);assert_eq!(estimate.seconds,228.);assert_eq!(estimate.working_bytes,56);
}
