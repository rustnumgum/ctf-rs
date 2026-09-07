use ctf::{cost::Models, mapping::{Distribution,Mapping,Topology}, node_reordering::{original_peer_counts,select_dense}};

#[test]
fn source_node_boundary_formula_preserves_fractional_peers() {
    assert_eq!(original_peer_counts(&[],1),Vec::<f64>::new());
    assert_eq!(original_peer_counts(&[2,2],1),vec![1.,1.]);
    assert_eq!(original_peer_counts(&[2,2],2),vec![0.,1.]);
    assert_eq!(original_peer_counts(&[2,2],4),vec![0.,0.]);
    assert_eq!(original_peer_counts(&[3,4],4),vec![0.5,2.]);
    assert_eq!(original_peer_counts(&[4,3],4),vec![0.,2.]);
}

#[test]
fn fractional_baseline_reaches_raw_tree_and_strict_node_selection() {
    let topology=Topology::new(vec![3,4]);
    let p=|axis|Mapping::Physical{axis,processes:topology.dimensions[axis],child:Box::new(Mapping::Unmapped)};
    let mapped=[Distribution::new(vec![12],topology.clone(),vec![p(0)]),
        Distribution::new(vec![12,8],topology.clone(),vec![p(0),p(1)]),
        Distribution::new(vec![8],topology.clone(),vec![p(1)])];
    let peers=original_peer_counts(&topology.dimensions,4);
    let models=Models::upstream(1);
    let choice=select_dense(mapped.each_ref(),["i","ij","j"],&peers,4,8,false,&models).unwrap();
    assert_eq!(choice.original_volume,72.);
    assert_eq!(choice.selected_volume,32.);
    assert_eq!(choice.inter_node_lens,vec![3,1]);
    assert_eq!(choice.intra_node_lens,vec![1,4]);
    assert!(select_dense(mapped.each_ref(),["i","ij","j"],&[2.,0.],4,8,false,&models).is_none());
}
