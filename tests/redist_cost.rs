use ctf::{cost::Models,mapping::{Distribution,Mapping,Topology},planning::GridPlan,redist_cost};
fn models()->Models{
    let mut m=Models::upstream(1);
    for name in ["seq_tsr_ctr_mdl_ref","bcast_mdl","red_mdl","red_mdl_cst","dgtog_res_mdl"]{
        m.get_mut(name).set_coefficients(&[0.,0.,1.]);
    }
    m.get_mut("blres_mdl").set_coefficients(&[0.,1.]);m
}
#[test]
fn unchanged_and_equal_phase_block_reshuffle(){
    let models=models();let topology=Topology::new(vec![2,2]);
    let mut i=Mapping::Unmapped;let mut j=Mapping::Unmapped;
    i.augment_physical(&topology,0);j.augment_physical(&topology,1);
    let old=Distribution::new(vec![3,5],topology.clone(),vec![i.clone(),j.clone()]);
    let new=Distribution::new(vec![3,5],topology,vec![j,i]);
    assert!(redist_cost::same_mapping(&old,&old));
    let unchanged=redist_cost::dense(&old,&old,8,&models);assert_eq!(unchanged.seconds,0.);assert_eq!(unchanged.temporary_bytes,0);
    assert!(redist_cost::can_block_reshuffle(&old,&new));
    let cost=redist_cost::dense(&old,&new,8,&models);assert_eq!(cost.seconds,48.);assert_eq!(cost.temporary_bytes,48);
}
#[test]
fn unfolded_total_keeps_source_output_roundtrip_and_memory_rule(){
    let models=models();
    for(shape,seconds,resident,temporary,memory)in[(vec![2],520.,192,408,600),(vec![2,2],636.,96,252,348)]{
        let topology=Topology::new(shape);let np=topology.size();
        let a=Distribution::cyclic(vec![3,4],np);let b=Distribution::cyclic(vec![4,5],np);
        let c=Distribution::cyclic(vec![3,5],np);
        let plan=GridPlan::prepare([&a,&b,&c],["ik","kj","ij"],topology.clone()).unwrap();
        let estimate=plan.estimate_unfolded(&models,8,&vec![1;topology.dimensions.len()],false);
        assert_eq!(estimate.seconds,seconds);assert_eq!(estimate.redistributed_input_bytes,resident);
        assert_eq!(estimate.redistribution_temporary_bytes,temporary);assert_eq!(estimate.memory_bytes,memory);
    }
}
