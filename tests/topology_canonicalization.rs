use ctf::{mapping::{Distribution,Mapping,Topology},mapping_variants::VariantSpace,mapping_preflight};
fn chain(t:&Topology,first:usize,second:usize)->Mapping{
    Mapping::Physical{axis:first,processes:t.dimensions[first],child:Box::new(Mapping::Physical{
        axis:second,processes:t.dimensions[second],child:Box::new(Mapping::Unmapped)})}
}
#[test]
fn folded_pair_reorders_catalog_and_physical_axes(){
    let t=Topology::new(vec![2,3,5]);
    let mut variant=VariantSpace::new([&[10],&[10],&[10]],["i","i","i"],Topology::new(vec![30]))
        .unwrap().decode(0).unwrap();
    variant.distributions=std::array::from_fn(|_|Distribution::new(vec![10],t.clone(),vec![chain(&t,2,0)]));
    let catalog=[t,Topology::new(vec![5,2,3])];
    assert_eq!(variant.canonicalize(&catalog),Some(1));
    assert!(variant.distributions.iter().all(|d|d.topology==catalog[1]));
    assert!(mapping_preflight::check(variant.distributions.each_ref(),["i","i","i"]));
}
#[test]
fn conflicting_pair_rejects_without_mutating_maps(){
    let t=Topology::new(vec![2,3,5]);
    let mut variant=VariantSpace::new([&[10],&[10],&[10]],["i","i","i"],Topology::new(vec![30]))
        .unwrap().decode(0).unwrap();
    variant.distributions=std::array::from_fn(|operand|Distribution::new(vec![10],t.clone(),
        vec![if operand==1{chain(&t,0,2)}else{chain(&t,2,0)}]));
    let old=variant.distributions.clone();
    assert_eq!(variant.canonicalize(&[t,Topology::new(vec![5,2,3])]),None);
    assert_eq!(variant.distributions,old);
}
