use ctf::{cost::Models,plan_cost::{Tree,Collective}};
fn models()->Models {
    let mut m=Models::upstream(1);
    for name in ["bcast_mdl","red_mdl","red_mdl_cst","seq_tsr_ctr_mdl_ref"] {
        m.get_mut(name).set_coefficients(&[1.,0.,0.]);
    }
    m
}
fn leaf()->Tree {Tree::Local{custom:false,folded:false,operand_bytes:[8,16,24],flops:12.}}
#[test]
fn recursion_and_layers() {
    let tree=Tree::Virtual{phases:vec![2,3],orders:[2,2,2],child:Box::new(Tree::Panels{
        steps:4,panel_bytes:[8,16,24],movement:[Some(Collective{ranks:2,nodes:2,bytes:8}),None,
            Some(Collective{ranks:4,nodes:3,bytes:24})],custom_reduce:false,child:Box::new(leaf())})};
    let m=models();
    for (layers,seconds,volume) in [(1,72.,0.),(2,36.,0.),(8,18.,0.)] {
        let e=tree.estimate(&m,layers);assert_eq!(e.seconds,seconds);assert_eq!(e.internode_volume,volume);
        assert_eq!(e.working_bytes,128); // 48 panel + 24 auxiliary + 56 virtual
    }
}
#[test]
fn replicas_and_nested_panels() {
    let tree=Tree::Replicated{inputs:[vec![Collective{ranks:2,nodes:2,bytes:8}],vec![]],
        output:vec![Collective{ranks:4,nodes:3,bytes:16}],custom_reduce:true,child:Box::new(leaf())};
    let m=models();let e=tree.estimate(&m,4);assert_eq!(e.seconds,3.);assert_eq!(e.working_bytes,0);assert_eq!(e.internode_volume,64.);
    let tree=Tree::Panels{steps:4,panel_bytes:[8,0,0],movement:[None,None,None],custom_reduce:false,
        child:Box::new(Tree::Panels{steps:2,panel_bytes:[8,0,0],movement:[None,None,None],custom_reduce:false,child:Box::new(leaf())})};
    let e=tree.estimate(&m,2);assert_eq!(e.seconds,4.);assert_eq!(e.working_bytes,16);
}
