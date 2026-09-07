use ctf::{cost::Models,mapping::{Distribution,Topology},planning::GridPlan};
fn model()->Models{
    let mut models=Models::upstream(1);
    for name in ["seq_tsr_ctr_mdl_ref","bcast_mdl","red_mdl","red_mdl_cst"]{
        models.get_mut(name).set_coefficients(&[0.,0.,1.]);
    }
    models
}
#[test]
fn generated_execution_tree_uses_source_local_flops_and_fiber_bytes(){
    let models=model();
    for (shape,seconds,volume) in [(vec![],120.,0.),(vec![2],168.,96.),(vec![2,2],156.,192.)]{
        let topology=Topology::new(shape);let np=topology.size();
        let a=Distribution::cyclic(vec![3,4],np);let b=Distribution::cyclic(vec![4,5],np);
        let c=Distribution::cyclic(vec![3,5],np);
        let plan=GridPlan::prepare([&a,&b,&c],["ik","kj","ij"],topology.clone()).unwrap();
        let nodes=if np==4{vec![1,2]}else{vec![1;topology.dimensions.len()]};
        let tree=plan.cost_tree(8,&nodes,false);
        let estimate=tree.estimate(&models,1);
        assert_eq!(estimate.seconds,seconds);
        assert_eq!(estimate.internode_volume,volume);
        assert_eq!(estimate.working_bytes,0);
    }
}
